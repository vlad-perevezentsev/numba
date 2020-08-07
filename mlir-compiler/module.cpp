#include <pybind11/pybind11.h>

#include "lowering.hpp"

namespace py = pybind11;

namespace
{
void lower_normal_function(py::object compilation_context, py::object func_ir)
{
    lower_function(compilation_context, func_ir);
}
}

PYBIND11_MODULE(mlir_compiler, m)
{
    m.def("lower_normal_function", &lower_normal_function, "todo");
}
