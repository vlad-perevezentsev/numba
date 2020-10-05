#pragma once

namespace pybind11
{
class bytes;
class object;
}

pybind11::bytes lower_function(const pybind11::object& compilation_context,
                               const pybind11::object& func_ir);
