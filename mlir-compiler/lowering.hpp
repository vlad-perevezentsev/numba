#pragma once

#include <pybind11/pybind11.h>

void lower_function(const pybind11::object& compilation_context, const pybind11::object& func_ir);
