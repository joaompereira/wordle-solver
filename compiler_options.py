NUMBA_COMPILER = True
# import NUMBA
try:
    if not NUMBA_COMPILER:
        raise ModuleNotFoundError
    from numba import njit, prange, generated_jit

    compiler_decorator = njit

except ModuleNotFoundError:
    def compiler_decorator(*args, **kwargs):
        def compiler_decorator_inner(fun):
            return fun

        return compiler_decorator_inner

    generated_jit = compiler_decorator

    prange = range

    NUMBA_COMPILER = False