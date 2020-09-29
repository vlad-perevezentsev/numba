#include "../_pymodule.h"
#include "../core/runtime/nrt_external.h"
#include "assert.h"
#include <dlfcn.h>
#include <stdio.h>

NRT_ExternalAllocator dparray_allocator;

void dparray_memsys_init() {
    void *(*get_queue)();
    void *sycldl = dlopen("libDPPLSyclInterface.so", RTLD_NOW);
    assert(sycldl != NULL);
    dparray_allocator.malloc = (NRT_external_malloc_func)dlsym(sycldl, "DPPLmalloc_shared");
    dparray_allocator.realloc = NULL;
    dparray_allocator.free = (NRT_external_free_func)dlsym(sycldl, "DPPLfree");
    get_queue = (void *(*))dlsym(sycldl, "DPPLGetCurrentQueue");
    dparray_allocator.opaque_data = get_queue();
//    printf("dparray_memsys_init: %p %p %p\n", dparray_allocator.malloc, dparray_allocator.free, dparray_allocator.opaque_data);
}

void * dparray_get_ext_allocator() {
    printf("dparray_get_ext_allocator %p\n", &dparray_allocator);
    return (void*)&dparray_allocator;
}

static PyObject *
get_external_allocator(PyObject *self, PyObject *args) {
    return PyLong_FromVoidPtr(dparray_get_ext_allocator());
}

static PyMethodDef ext_methods[] = {
#define declmethod_noargs(func) { #func , ( PyCFunction )func , METH_NOARGS, NULL }
    declmethod_noargs(get_external_allocator),
    {NULL},
#undef declmethod_noargs
};

static PyObject *
build_c_helpers_dict(void)
{
    PyObject *dct = PyDict_New();
    if (dct == NULL)
        goto error;

#define _declpointer(name, value) do {                 \
    PyObject *o = PyLong_FromVoidPtr(value);           \
    if (o == NULL) goto error;                         \
    if (PyDict_SetItemString(dct, name, o)) {          \
        Py_DECREF(o);                                  \
        goto error;                                    \
    }                                                  \
    Py_DECREF(o);                                      \
} while (0)

    _declpointer("dparray_get_ext_allocator", &dparray_get_ext_allocator);

#undef _declpointer
    return dct;
error:
    Py_XDECREF(dct);
    return NULL;
}

MOD_INIT(_dppl_rt) {
    PyObject *m;
    MOD_DEF(m, "numba.dppl._dppl_rt", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;
    dparray_memsys_init();
    PyModule_AddObject(m, "c_helpers", build_c_helpers_dict());
    return MOD_SUCCESS_VAL(m);
}
