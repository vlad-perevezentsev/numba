from __future__ import print_function, division, absolute_import

import numba
import numba.dppl.dparray as np
import numpy

def p1(a):
    return a * 2.0 + 13

f1 = numba.njit(p1)

@numba.njit()
def f2(a):
    return a

@numba.njit()
def f3(a, b):
    return a * np.ndarray(b.shape, b.dtype, b)

@numba.njit()
def f4():
    return np.ones(10)

def p5(a, b):
    return a * b

f5 = numba.njit(p5)

@numba.njit()
def f6(a):
    return a + 13

print("Testing Python Numpy")
z1 = numpy.ones(10)
z2 = p1(z1)
print("z2:", z2, type(z2))
assert(isinstance(z2, numpy.ndarray))

print("Testing Numba Numpy")
z1 = numpy.ones(10)
z2 = f1(z1)
print("z2:", z2, type(z2))
assert(isinstance(z2, numpy.ndarray))

print("Testing dparray ones")
a = np.ones(10)
print("a:", a, type(a))
assert(isinstance(a, np.ndarray))

print("Testing dparray multiplication")
c = a * 5
print("c", c, type(c))
assert(isinstance(c, np.ndarray))

print("Testing Python dparray")
b = p1(c)
print("b:", b, type(b))
assert(isinstance(b, np.ndarray))

print("Testing Numba dparray")
b = f1(c)
print("b:", b, type(b))
assert(isinstance(b, np.ndarray))

print("Testing Numba dparray 2")
d = f2(a)
print("d:", d, type(d))
assert(isinstance(b, np.ndarray))

print("Testing Numba dparray constructor from numpy.ndarray")
e = f3(a, z1)
print("e:", e, type(e))
assert(isinstance(e, np.ndarray))

print("Testing Numba dparray functions")
f = f4()
print("f:", f, type(f))
assert(isinstance(f, np.ndarray))

print("Testing Python mixing dparray and numpy.ndarray")
h = p5(a, z1)
print("h:", h, type(h))
assert(isinstance(h, np.ndarray))

print("Testing Numba mixing dparray and numpy.ndarray")
h = f5(a, z1)
print("h:", h, type(h))
assert(isinstance(h, np.ndarray))

print("Testing Numba mixing dparray and constant")
g = f6(a)
print("g:", g, type(g))
assert(isinstance(g, np.ndarray))
