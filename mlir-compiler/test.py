import numba

@numba.njit
def sum1(a):
    return a + 42

def sum2(a, b):
    return a + b

@numba.njit
def cond(a, b):
    if a > b:
        return a
    else:
        return b

def test(func, result, params):
    print('test', func.__name__, params, '... ', end='')
    try:
        res = func(*params)
        if (res != result):
            raise Exception(f'Invalid value "{res}", expected "{result}"')
        print('SUCCESS')
    except Exception as e:
        print(e)
        print('FAILED')
    

test(sum1, 47, (5,))
test(sum2, 7, (3,4))
test(cond, 6, (5,6))
test(cond, 8, (8,7))
