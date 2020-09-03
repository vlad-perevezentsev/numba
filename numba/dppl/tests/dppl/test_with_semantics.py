from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase
from numba.dppl.withcontexts import dppl_context
from numba.core import typing, cpu
from numba.core.compiler import compile_ir, DEFAULT_FLAGS
from numba.core.transforms import with_lifting
from numba.core.registry import cpu_target
from numba.core.bytecode import FunctionIdentity, ByteCode
from numba.core.interpreter import Interpreter
from numba.tests.support import captured_stdout
from numba import njit, prange
import numpy as np


def get_func_ir(func):
    func_id = FunctionIdentity.from_function(func)
    bc = ByteCode(func_id=func_id)
    interp = Interpreter(func_id)
    func_ir = interp.interpret(bc)
    return func_ir


def liftcall1():
    x = 1
    print("A", x)
    with dppl_context:
        x += 1
    print("B", x)
    return x


def liftcall2():
    x = 1
    print("A", x)
    with dppl_context:
        x += 1
    print("B", x)
    with dppl_context:
        x += 10
    print("C", x)
    return x


def liftcall3():
    x = 1
    print("A", x)
    with dppl_context:
        if x > 0:
            x += 1
    print("B", x)
    with dppl_context:
        for i in range(10):
            x += i
    print("C", x)
    return x


class BaseTestWithLifting(DPPLTestCase):
    def setUp(self):
        super(BaseTestWithLifting, self).setUp()
        self.typingctx = typing.Context()
        self.targetctx = cpu.CPUContext(self.typingctx)
        self.flags = DEFAULT_FLAGS

    def check_extracted_with(self, func, expect_count, expected_stdout):
        the_ir = get_func_ir(func)
        new_ir, extracted = with_lifting(
            the_ir, self.typingctx, self.targetctx, self.flags,
            locals={},
        )
        self.assertEqual(len(extracted), expect_count)
        cres = self.compile_ir(new_ir)

        with captured_stdout() as out:
            cres.entry_point()

        self.assertEqual(out.getvalue(), expected_stdout)

    def compile_ir(self, the_ir, args=(), return_type=None):
        typingctx = self.typingctx
        targetctx = self.targetctx
        flags = self.flags
        # Register the contexts in case for nested @jit or @overload calls
        with cpu_target.nested_context(typingctx, targetctx):
            return compile_ir(typingctx, targetctx, the_ir, args,
                              return_type, flags, locals={})


class TestLiftCall(BaseTestWithLifting):

    def check_same_semantic(self, func):
        """Ensure same semantic with non-jitted code
        """
        jitted = njit(target="gpu")(func)
        with captured_stdout() as got:
            jitted()

        with captured_stdout() as expect:
            func()

        self.assertEqual(got.getvalue(), expect.getvalue())

    def test_liftcall1(self):
        self.check_extracted_with(liftcall1, expect_count=1,
                                  expected_stdout="A 1\nB 2\n")
        self.check_same_semantic(liftcall1)

    def test_liftcall2(self):
        self.check_extracted_with(liftcall2, expect_count=2,
                                  expected_stdout="A 1\nB 2\nC 12\n")
        self.check_same_semantic(liftcall2)

    def test_liftcall3(self):
        self.check_extracted_with(liftcall3, expect_count=2,
                                  expected_stdout="A 1\nB 2\nC 47\n")
        self.check_same_semantic(liftcall3)


if __name__ == '__main__':
    unittest.main()
