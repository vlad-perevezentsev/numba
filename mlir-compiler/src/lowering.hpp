#pragma once

namespace pybind11
{
class bytes;
class capsule;
class object;
}

pybind11::capsule create_module();

pybind11::capsule lower_function(const pybind11::object& compilation_context,
                                 const pybind11::capsule& py_mod,
                                 const pybind11::object& func_ir);

pybind11::bytes serialize_module(const pybind11::capsule& py_mod);
