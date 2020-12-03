#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Function.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "plier/PlierOpsEnums.h.inc"
#include "plier/PlierOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "plier/PlierOps.h.inc"

namespace plier
{
namespace attributes
{
const constexpr llvm::StringLiteral fastmath("#plier.fastmath");
const constexpr llvm::StringLiteral jump_markers("#plier.pipeline_jump_markers");
}

namespace detail
{
struct PyTypeStorage;
}

class PyType : public mlir::Type::TypeBase<::plier::PyType, mlir::Type,
                                           ::plier::detail::PyTypeStorage>
{
public:
    using Base::Base;

    static PyType get(mlir::MLIRContext *context, mlir::StringRef name);
    static PyType getUndefined(mlir::MLIRContext *context);
    static PyType getNone(mlir::MLIRContext *context);

    mlir::StringRef getName() const;
};


}
