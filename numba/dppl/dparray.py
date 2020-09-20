#from ._ndarray_utils import _transmogrify
import numpy as np
from inspect import getmembers, isfunction, isclass
from numbers import Number
import numba
from numba import types
from numba.extending import typeof_impl, register_model, type_callable, lower_builtin
from numba.np import numpy_support
from numba.core.pythonapi import box, allocator
from llvmlite import ir
import llvmlite.binding as llb
from numba.core import types, cgutils
import builtins
import sys
from ctypes.util import find_library
import dppl
from dppl._memory import MemoryUSMShared

flib = find_library('mkl_intel_ilp64')
print("flib:", flib)
llb.load_library_permanently(flib)

functions_list = [o for o in getmembers(np) if isfunction(o[1])]
class_list = [o for o in getmembers(np) if isclass(o[1])]

class ndarray(np.ndarray):
    """
    numpy.ndarray subclass whose underlying memory buffer is allocated
    with a foreign allocator.
    """
    def __new__(subtype, shape,
                dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        # Create a new array.
        if buffer is None:
            nelems = np.prod(shape)
            dt = np.dtype(dtype)
            isz = dt.itemsize
            buf = MemoryUSMShared(nbytes=isz*max(1,nelems))
            return np.ndarray.__new__(
                subtype, shape, dtype=dt,
                buffer=buf, offset=0,
                strides=strides, order=order)
        # zero copy if buffer is a usm backed array-like thing
        elif hasattr(buffer, '__sycl_usm_array_interface__'):
            # also check for array interface
            return np.ndarray.__new__(
                subtype, shape, dtype=dt,
                buffer=buffer, offset=offset,
                strides=strides, order=order)
        else:
            # must copy
            ar = np.ndarray(shape,
                            dtype=dtype, buffer=buffer,
                            offset=offset, strides=strides,
                            order=order)
            buf = MemoryUSMShared(nbytes=ar.nbytes)
            res = np.ndarray.__new__(
                subtype, shape, dtype=dtype,
                buffer=buf, offset=0,
                strides=strides, order=order)
            np.copyto(res, ar, casting='no')
            return res

    def __array_finalize__(self, obj):
        # When called from the explicit constructor, obj is None
        if obj is None: return
        # When called in new-from-template, `obj` is another instance of our own
        # subclass, that we might use to update the new `self` instance.
        # However, when called from view casting, `obj` can be an instance of any
        # subclass of ndarray, including our own.
        if hasattr(obj, '__sycl_usm_array_interface__'):
            return
        if isinstance(obj, np.ndarray):
            ob = self
            while isinstance(ob, np.ndarray):
                if hasattr(obj, '__sycl_usm_array_interface__'):
                    return
                ob = ob.base
    
            # trace if self has underlying mkl_mem buffer
#            ob = self.base
           
#            while isinstance(ob, ndarray):
#                ob = ob.base
#            if isinstance(ob, dppl.Memory):
#                return

        # Just raise an exception since __array_ufunc__ makes all reasonable cases not
        # need the code below.
        raise ValueError("Non-MKL allocated ndarray can not viewed as MKL-allocated one without a copy")
      
        """
        # since dparray must have mkl_memory underlying it, a copy must be made
        newbuf = dppl.Memory(nbytes=self.data.nbytes)
        new_arr = np.ndarray.__new__(
            type(self),
            self.shape,
            buffer=newbuf, offset=0,
            dtype=self.dtype,
            strides=self.strides)
        np.copyto(new_arr, self)
        # We need to modify self to now be mkl_memory-backed ndarray
        # We only need to change data and base, but these are not writeable.
        #
        # Modification can not be done by simply setting self either,
        # as self is just a local copy of the instance.
        #
        # raise ValueError("Non-MKL allocated ndarray can not viewed as MKL-allocated one without a copy")
        # Will probably have to raise an exception soon as Numpy may disallow this.
        _transmogrify(self, new_arr)
        """

    __numba_no_subtype_ndarray__ = True

    def from_ndarray(x):
        return ndarray(x.shape, x.dtype, x)

    def as_ndarray(self):
        return np.ndarray(self.shape, self.dtype, self)

    def __array__(self):
        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            N = None
            scalars = []
            for inp in inputs:
                if isinstance(inp, Number):
                    scalars.append(inp)
                elif isinstance(inp, (self.__class__, np.ndarray)):
                    if isinstance(inp, self.__class__):
                        scalars.append(np.ndarray(inp.shape, inp.dtype, inp))
                    else:
                        scalars.append(inp)
                    if N is not None:
                        if N != inp.shape:
                            raise TypeError("inconsistent sizes")
                    else:
                        N = inp.shape
                else:
                    return NotImplemented
            if kwargs.get('out', None) is None:
                # maybe copy?
                # deal with multiple returned arrays, so kwargs['out'] can be tuple
                kwargs['out'] = empty(inputs[0].shape, dtype=get_ret_type_from_ufunc(ufunc))
            ret = ufunc(*scalars, **kwargs)
            return ret
#            return self.__class__(ret.shape, ret.dtype, ret)
        else:
            return NotImplemented

for c in class_list:
    cname = c[0]
    new_func =  "class %s(np.%s):\n" % (cname, cname)
    if cname == "ndarray":
        # Implemented explicitly above.
        continue
    else:
        # This is temporary.
        new_func += "    pass\n"
        # The code below should eventually be made to work and used.
#        new_func += "    @classmethod\n"
#        new_func += "    def cast(cls, some_np_obj):\n"
#        new_func += "        some_np_obj.__class__ = cls\n"
#        new_func += "        return some_np_obj\n"
    try:
        the_code = compile(new_func, '__init__', 'exec')
        exec(the_code)
    except:
        pass

# Redefine all Numpy functions in this module and if they
# return a Numpy array, transform that to a USM-backed array
# instead.  This is a stop-gap.  We should eventually find a
# way to do the allocation correct to start with.
for f in functions_list:
    fname = f[0]
    new_func =  "def %s(*args, **kwargs):\n" % fname
    new_func += "    ret = np.%s(*args, **kwargs)\n" % fname
    new_func += "    if type(ret) == np.ndarray:\n"
    new_func += "        ret = ndarray(ret.shape, ret.dtype, ret)\n"
    new_func += "    return ret\n"
    the_code = compile(new_func, '__init__', 'exec')
    exec(the_code)

# This class creates a type in Numba.
class DPArrayType(types.Array):
    def __init__(self, dtype, ndim, layout, readonly=False, name=None,
                 aligned=True, addrspace=None):
        # This name defines how this type will be shown in Numba's type dumps.
        name = "DPArray:ndarray(%s, %sd, %s)" % (dtype, ndim, layout)
        super(DPArrayType, self).__init__(dtype, ndim, layout,
                                            py_type=ndarray,
                                            readonly=readonly,
                                            name=name,
                                            addrspace=addrspace)

    # Tell Numba typing how to combine DPArrayType with other ndarray types.
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            for inp in inputs:
                if not isinstance(inp, (DPArrayType, types.Array, types.Number)):
                    return None

            return DPArrayType
        else:
            return None

# This tells Numba how to create a DPArrayType when a dparray is passed
# into a njit function.
@typeof_impl.register(ndarray)
def typeof_ta_ndarray(val, c):
    try:
        dtype = numpy_support.from_dtype(val.dtype)
    except NotImplementedError:
        raise ValueError("Unsupported array dtype: %s" % (val.dtype,))
    layout = numpy_support.map_layout(val)
    readonly = not val.flags.writeable
    return DPArrayType(dtype, val.ndim, layout, readonly=readonly)

# This tells Numba to use the default Numpy ndarray data layout for
# object of type DPArray.
register_model(DPArrayType)(numba.core.datamodel.models.ArrayModel)

# This tells Numba how to type calls to a DPArray constructor.
@type_callable(ndarray)
def type_ndarray(context):
    def typer(shape, ndim, buf):
        return DPArrayType(buf.dtype, buf.ndim, buf.layout)
    return typer

# This tells Numba how to implement calls to a DPArray constructor.
@lower_builtin(ndarray, types.UniTuple, types.DType, types.Array)
def impl_ndarray(context, builder, sig, args):
    # Need to allocate and copy here!
    shape, ndim, buf = args
    return buf

# This tells Numba how to convert from its native representation
# of a DPArray in a njit function back to a Python DPArray.
@box(DPArrayType)
def box_array(typ, val, c):
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder, value=val)
    if c.context.enable_nrt:
        np_dtype = numpy_support.as_dtype(typ.dtype)
        dtypeptr = c.env_manager.read_const(c.env_manager.add_const(np_dtype))
        # Steals NRT ref
        newary = c.pyapi.nrt_adapt_ndarray_to_python(typ, val, dtypeptr)
        return newary
    else:
        parent = nativeary.parent
        c.pyapi.incref(parent)
        return parent

# This tells Numba to use this function when it needs to allocate a
# DPArray in a njit function.
@allocator(DPArrayType)
def allocator_DPArray(context, builder, size, align):
    print("allocator_DPArray")
    sys.stdout.flush()
    use_Numba_allocator = True
    if use_Numba_allocator:
        print("Using Numba allocator")
        context.nrt._require_nrt()

        mod = builder.module
        u32 = ir.IntType(32)
        fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t, u32])
        fn = mod.get_or_insert_function(fnty,
                                        name="NRT_MemInfo_alloc_safe_aligned")
        fn.return_value.add_attribute("noalias")
        if isinstance(align, builtins.int):
            align = context.get_constant(types.uint32, align)
        else:
            assert align.type == u32, "align must be a uint32"
        return builder.call(fn, [size, align])
    else:
        print("Using mkl_malloc")
        context.nrt._require_nrt()

        mod = builder.module
        u32 = ir.IntType(32)
        fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t, u32])
        fn = mod.get_or_insert_function(fnty, name="mkl_malloc")
        fn.return_value.add_attribute("noalias")
        if isinstance(align, builtins.int):
            align = context.get_constant(types.uint32, align)
        else:
            assert align.type == u32, "align must be a uint32"
        return builder.call(fn, [size, align])
