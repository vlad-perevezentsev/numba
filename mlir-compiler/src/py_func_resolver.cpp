#include "py_func_resolver.hpp"

#include <pybind11/pybind11.h>

#include <mlir/IR/BuiltinOps.h>

namespace py = pybind11;

namespace
{

template<unsigned Width, mlir::IntegerType::SignednessSemantics Signed>
bool is_int(mlir::Type type)
{
    if (auto t = type.dyn_cast<mlir::IntegerType>())
    {
        if (t.getWidth() == Width && t.getSignedness() == Signed)
        {
            return true;
        }
    }
    return false;
}

template<unsigned Width>
bool is_float(mlir::Type type)
{
    if (auto f = type.dyn_cast<mlir::FloatType>())
    {
        if (f.getWidth() == Width)
        {
            return true;
        }
    }
    return false;
}

py::handle map_type(const py::handle& types_mod, mlir::Type type)
{
    using fptr_t = bool(*)(mlir::Type);
    const std::pair<fptr_t, llvm::StringRef> primitive_types[] = {
        {&is_int<1, mlir::IntegerType::Signed>,   "boolean"},
        {&is_int<1, mlir::IntegerType::Signless>, "boolean"},
        {&is_int<1, mlir::IntegerType::Unsigned>, "boolean"},

        {&is_int<8, mlir::IntegerType::Signed>,    "int8"},
        {&is_int<8, mlir::IntegerType::Signless>,  "int8"},
        {&is_int<8, mlir::IntegerType::Unsigned>, "uint8"},

        {&is_int<16, mlir::IntegerType::Signed>,    "int16"},
        {&is_int<16, mlir::IntegerType::Signless>,  "int16"},
        {&is_int<16, mlir::IntegerType::Unsigned>, "uint16"},

        {&is_int<32, mlir::IntegerType::Signed>,    "int32"},
        {&is_int<32, mlir::IntegerType::Signless>,  "int32"},
        {&is_int<32, mlir::IntegerType::Unsigned>, "uint32"},

        {&is_int<64, mlir::IntegerType::Signed>,    "int64"},
        {&is_int<64, mlir::IntegerType::Signless>,  "int64"},
        {&is_int<64, mlir::IntegerType::Unsigned>, "uint64"},

        {&is_float<32>, "float"},
        {&is_float<64>, "double"},
    };

    for (auto h : primitive_types)
    {
        if (h.first(type))
        {
            auto name = h.second;
            return types_mod.attr(py::str(name.data(), name.size()));
        }
    }

    if (auto m = type.dyn_cast<mlir::MemRefType>())
    {
        auto elem_type = map_type(types_mod, m.getElementType());
        if (!elem_type)
        {
            return {};
        }
        auto ndims = py::int_(m.getRank());
        auto array_type = types_mod.attr("Array");
        return array_type(elem_type, ndims, py::str("C"));
    }
    return {};
}

py::list map_types(const py::handle& types_mod, mlir::TypeRange types)
{
    py::list ret;
    for (auto type : types)
    {
        auto elem = map_type(types_mod, type);
        if (!elem)
        {
            return py::none();
        }
        ret.append(std::move(elem));
    }
    return ret;
}
}

struct PyFuncResolver::Context
{
    py::handle resolver;
    py::handle compiler;
    py::handle types;
};

PyFuncResolver::PyFuncResolver():
    context(std::make_unique<Context>())
{
    auto registry_mod = py::module::import("numba.mlir.func_registry");
    auto compiler_mod = py::module::import("numba.mlir.inner_compiler");
    context->resolver = registry_mod.attr("find_active_func");
    context->compiler = compiler_mod.attr("compile_func");
    context->types = py::module::import("numba.core.types");
}

PyFuncResolver::~PyFuncResolver()
{

}

mlir::FuncOp PyFuncResolver::get_func(llvm::StringRef name, mlir::TypeRange types)
{
    assert(!name.empty());
    auto py_name = py::str(name.data(), name.size());
    auto py_func = context->resolver(py_name);
    if (py_func.is_none())
    {
        return {};
    }
    auto py_types = map_types(context->types, types);
    if (py_types.is_none())
    {
        return {};
    }
    auto res = static_cast<mlir::Operation*>(context->compiler(py_func, py_types).cast<py::capsule>());
    auto func = (res ? mlir::cast<mlir::FuncOp>(res) : nullptr);
    return func;
}
