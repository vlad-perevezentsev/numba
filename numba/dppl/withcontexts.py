from numba.core import compiler, typing, types, sigutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.dispatcher import _DispatcherBase
from numba.core.transforms import find_region_inout_vars
from numba.core.withcontexts import (WithContext, _mutate_with_block_callee, _mutate_with_block_caller,
                                     _clear_blocks)
from numba.dppl.compiler import DPPLCompiler
from numba.core.cpu_options import ParallelOptions


class _DPPLContextType(WithContext):
    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end,
                         body_blocks, dispatcher_factory, extra):
        assert extra is None
        vlt = func_ir.variable_lifetime

        inputs, outputs = find_region_inout_vars(
            blocks=blocks,
            livemap=vlt.livemap,
            callfrom=blk_start,
            returnto=blk_end,
            body_block_ids=set(body_blocks),
            )

        lifted_blks = {k: blocks[k] for k in body_blocks}
        _mutate_with_block_callee(lifted_blks, blk_start, blk_end,
                                  inputs, outputs)

        # XXX: transform body-blocks to return the output variables
        lifted_ir = func_ir.derive(
            blocks=lifted_blks,
            arg_names=tuple(inputs),
            arg_count=len(inputs),
            force_non_generator=True,
            )

        dispatcher = dispatcher_factory(lifted_ir, dppl_mode=True)
        
        newblk = _mutate_with_block_caller(
            dispatcher, blocks, blk_start, blk_end, inputs, outputs,
            )

        blocks[blk_start] = newblk
        _clear_blocks(blocks, body_blocks)
        return dispatcher


class DPPLLiftedCode(_DispatcherBase):
    """
    Implementation of the hidden dispatcher objects used for lifted code
    (a lifted loop is really compiled as a separate function).
    """
    _fold_args = False

    def __init__(self, func_ir, typingctx, targetctx, flags, locals):
        self.func_ir = func_ir
        self.lifted_from = None

        self.typingctx = typingctx
        self.targetctx = targetctx
        self.flags = flags
        self.locals = locals

        _DispatcherBase.__init__(self, self.func_ir.arg_count,
                                 self.func_ir.func_id.func,
                                 self.func_ir.func_id.pysig,
                                 can_fallback=True,
                                 exact_match_required=False)

    def get_source_location(self):
        """Return the starting line number of the loop.
        """
        return self.func_ir.loc.line

    def _pre_compile(self, args, return_type, flags):
        """Pre-compile actions
        """
        pass

    @global_compiler_lock
    def compile(self, sig):
        # Use counter to track recursion compilation depth
        with self._compiling_counter:
            # XXX this is mostly duplicated from Dispatcher.
            flags = self.flags
            args, return_type = sigutils.normalize_signature(sig)

            # Don't recompile if signature already exists
            # (e.g. if another thread compiled it before we got the lock)
            existing = self.overloads.get(tuple(args))
            if existing is not None:
                return existing.entry_point

            self._pre_compile(args, return_type, flags)

            # Clone IR to avoid (some of the) mutation in the rewrite pass
            cloned_func_ir = self.func_ir.copy()

            flags.auto_parallel = ParallelOptions({'offload':True})
            cres = compiler.compile_ir(typingctx=self.typingctx,
                                       targetctx=self.targetctx,
                                       func_ir=cloned_func_ir,
                                       args=args, return_type=return_type,
                                       flags=flags, locals=self.locals,
                                       lifted=(),
                                       lifted_from=self.lifted_from,
                                       is_lifted_loop=True,
                                       pipeline_class=DPPLCompiler)
            
            # Check typing error if object mode is used
            if cres.typing_error is not None and not flags.enable_pyobject:
                raise cres.typing_error
            self.add_overload(cres)
            return cres.entry_point


class DPPLLiftedWith(DPPLLiftedCode):
    @property
    def _numba_type_(self):
        return types.Dispatcher(self)

    def get_call_template(self, args, kws):
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This enables the resolving of the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
        # Ensure an overload is available
        if self._can_compile:
            self.compile(tuple(args))

        pysig = None
        # Create function type for typing
        func_name = self.py_func.__name__
        name = "CallTemplate({0})".format(func_name)
        # The `key` isn't really used except for diagnosis here,
        # so avoid keeping a reference to `cfunc`.
        call_template = typing.make_concrete_template(
            name, key=func_name, signatures=self.nopython_signatures)
        return call_template, pysig, args, kws


dppl_context = _DPPLContextType()
