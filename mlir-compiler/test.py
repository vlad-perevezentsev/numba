import numba

def sum1(a):
    return a + 42

def sum2(a, b):
    return a + b

def cond(a, b):
    if a > b:
        return a
    else:
        return b



def test(func, params):
    print('test', func.__name__, params, '... ', end='')
    result = func(*params)
    wrapped = numba.njit()(func)
    try:
        res = wrapped(*params)
        if (res != result):
            raise Exception(f'Invalid value "{res}", expected "{result}"')
        print('SUCCESS')
    except Exception as e:
        print(e)
        print('FAILED')
    

test(sum1, (5,))
test(sum2, (3,4))
test(cond, (5,6))
test(cond, (8,7))
