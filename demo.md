# Demo: Switchable target and validating target of nested functions

Branch: https://github.com/sklam/numba/tree/demo/extendtarget_demo_1

Based on PRs:
- https://github.com/numba/numba/pull/6814
- https://github.com/numba/numba/pull/6762
- https://github.com/numba/numba/pull/6870



## 1. Add target options that can be inherited by callee

Using feature in https://github.com/numba/numba/pull/6814

The commit https://github.com/sklam/numba/commit/073a91cbc00bf59475c62133ce73fac210892ef3 shows how a new target option is added for tracking the compilation target. The option is named `target_backend` in the code. It is exposed in the decorator (i.e. in `@jit`). When left unspecified, callees will inherit the value from the caller. If there are no caller (i.e. called from the interpreter), the default is "cpu".


## 2. Add legalization pass that scan for invalid target in callees.

The commit https://github.com/sklam/numba/commit/f30449336c3b8991a7cc6dbcd103583991f7d64b adds a legalization pass to validate the `target_backend` in callees. This pass will only run if the current `target_backend` is `"CustomCPU"`. It checks that the callee's `target_backend` matches to `"CustomCPU"`.

## 3. The Demo

The commit https://github.com/sklam/numba/commit/bdf32a6622c454cff2762771881aacc1e3a01975 adds the test cases.


Adds a `"CustomCPU"` target and a new dispatcher subclass:
https://github.com/sklam/numba/blob/bdf32a6622c454cff2762771881aacc1e3a01975/demo.py#L11-L20


Details of the demo file:
- Using features in https://github.com/numba/numba/pull/6870, adds the utils to switch target in a contextmanager:
https://github.com/sklam/numba/blob/bdf32a6622c454cff2762771881aacc1e3a01975/demo.py#L21-L39
- a function that is fixed to `"cpu"` target:
https://github.com/sklam/numba/blob/bdf32a6622c454cff2762771881aacc1e3a01975/demo.py#L44-L51
- a function that has flexible target but calls the previous fixed target function:
https://github.com/sklam/numba/blob/bdf32a6622c454cff2762771881aacc1e3a01975/demo.py#L54-L61
- a function that has flexible target:
https://github.com/sklam/numba/blob/bdf32a6622c454cff2762771881aacc1e3a01975/demo.py#L64-L71


The [`demo.py`](https://github.com/sklam/numba/blob/bdf32a6622c454cff2762771881aacc1e3a01975/demo.py) takes the test case number as a commandline argument; for example:

```bash
$ python demo.py 0  # runs test case 0
$ python demo.py 3  # runs test case 3
```

### Case 0

https://github.com/sklam/numba/blob/bdf32a6622c454cff2762771881aacc1e3a01975/demo.py#L77-L91

Normal execution of in the default `"cpu"` target.


### Case 1

https://github.com/sklam/numba/blob/bdf32a6622c454cff2762771881aacc1e3a01975/demo.py#L94-L107

Using `switch_target` to make `foo` run in `"CustomCPU"` target.

### Case 2

https://github.com/sklam/numba/blob/bdf32a6622c454cff2762771881aacc1e3a01975/demo.py#L110-L124

Similar to case 1 but calls `fixed_target` which is not allowed to run in `"CustomCPU"` target. This will print the following:

```bash
-----------------------------------Compiling------------------------------------
>> case2.<locals>.foo
   flags=Flags(enable_looplift=True, enable_pyobject=False, enable_pyobject_looplift=True, debuginfo=False, boundscheck=False, nrt=True, target_backend=CustomCPU)
-----------------------------------Compiling------------------------------------
>> fixed_target
   flags=Flags(enable_looplift=True, enable_pyobject=False, enable_pyobject_looplift=True, debuginfo=False, boundscheck=False, nrt=True, fastmath=FastMathOptions(set()), target_backend=cpu)
-----------------------------------Compiling------------------------------------
>> flex_target
   flags=Flags(enable_looplift=True, enable_pyobject=False, enable_pyobject_looplift=True, debuginfo=False, boundscheck=False, nrt=True, fastmath=FastMathOptions(set()), target_backend=CustomCPU)
Running LegalizeForTarget in flex_target
Running LegalizeForTarget in case2.<locals>.foo
   CALL $10load_global.3 :: type(CPUDispatcher(<function flex_target at 0x11cae4670>))
   CALL $2load_global.0 :: type(CPUDispatcher(<function fixed_target at 0x11cae44c0>))
Traceback (most recent call last):
  File "demo.py", line 182, in <module>
    main()
  ...
RuntimeError: Failed in nopython mode pipeline (step: legalize for target)
Backend Mismatch in type(CPUDispatcher(<function fixed_target at 0x11cae44c0>))!
```


### Case 3

https://github.com/sklam/numba/blob/bdf32a6622c454cff2762771881aacc1e3a01975/demo.py#L127-L141

Similar to case 2 but calls `flex_call_fixed`, which indirectly calls `fixed_target`. This will print the following:

```bash
$ python demo.py 3
-----------------------------------Compiling------------------------------------
>> case3.<locals>.foo
   flags=Flags(enable_looplift=True, enable_pyobject=False, enable_pyobject_looplift=True, debuginfo=False, boundscheck=False, nrt=True, target_backend=CustomCPU)
-----------------------------------Compiling------------------------------------
>> flex_call_fixed
   flags=Flags(enable_looplift=True, enable_pyobject=False, enable_pyobject_looplift=True, debuginfo=False, boundscheck=False, nrt=True, fastmath=FastMathOptions(set()), target_backend=CustomCPU)
-----------------------------------Compiling------------------------------------
>> fixed_target
   flags=Flags(enable_looplift=True, enable_pyobject=False, enable_pyobject_looplift=True, debuginfo=False, boundscheck=False, nrt=True, fastmath=FastMathOptions(set()), target_backend=cpu)
Running LegalizeForTarget in flex_call_fixed
   CALL $10load_global.3 :: type(CPUDispatcher(<function fixed_target at 0x122e9f4c0>))
-----------------------------------Compiling------------------------------------
>> flex_call_fixed
   flags=Flags(enable_looplift=True, enable_pyobject=False, enable_pyobject_looplift=True, debuginfo=False, boundscheck=False, nrt=True, fastmath=FastMathOptions(set()), target_backend=CustomCPU)
Running LegalizeForTarget in flex_call_fixed
   CALL $10load_global.3 :: type(CPUDispatcher(<function fixed_target at 0x122e9f4c0>))
Traceback (most recent call last):
  File "demo.py", line 182, in <module>
    main()
  ...
numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
Internal error at <numba.core.typeinfer.CallConstraint object at 0x123de0100>.
Failed in nopython mode pipeline (step: legalize for target)
Backend Mismatch in type(CPUDispatcher(<function fixed_target at 0x122e9f4c0>))!
During: resolving callee type: type(CPUDispatcher(<function flex_call_fixed at 0x122e9f280>))
During: typing of call at demo.py (135)

Enable logging at debug level for details.

File "demo.py", line 135:
    def foo(x):
        x = flex_call_fixed(x)  # calls fixed_target indirectly
        ^
```

Notice when compiling `flex_call_fixed`, the `target_backend` flag is inheriting the caller `case3.<locals>.foo` to obtain the value `CustomCPU`.


### Case 4

https://github.com/sklam/numba/blob/bdf32a6622c454cff2762771881aacc1e3a01975/demo.py#L144-L168

Similar to case 3, but `flex_call_fixed` is invoked for the `cpu` target before the invocation in a `switch_target` context. **The same error as in case 3 is expected but a known issue is preventing it.** The in-memory cache for the functions is using type-signature as key. We will need to add the target options as part of the key so it will not pick up a previous compilation result from a wrong target.
