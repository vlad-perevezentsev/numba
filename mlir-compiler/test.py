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

def var(a):
    c = 1
    c = c + a
    return c

def jump(a, b):
    c = 3
    if a > 5:
        c = c + a
    c = c + b
    return c

def loop(n):
    res = 0
    for i in range(n):
        res += i
    return res


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
test(var, (8,))
test(jump, (1,8))
test(jump, (7,8))
#test(loop, (8,))
