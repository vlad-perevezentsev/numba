import numba

@numba.njit
def sum1(a):
    return a + 42

def sum2(a, b):
    return a + b

@numba.njit
def bar(a, b):
    if a > b:
        return a
    else:
        return b

def test(func, result, params):
    print('test', func.__name__, '... ', end='')
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
#print(bar(5,6))