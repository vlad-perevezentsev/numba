#pragma once

#include <mlir/IR/Attributes.h>

namespace mlir
{
class Operation;
class Value;
}

mlir::Attribute getConstVal(mlir::Operation* op);
mlir::Attribute getConstVal(mlir::Value op);

template<typename T>
T getConstVal(mlir::Operation* op)
{
    return getConstVal(op).dyn_cast_or_null<T>();
}

template<typename T>
T getConstVal(mlir::Value op)
{
    return getConstVal(op).dyn_cast_or_null<T>();
}
