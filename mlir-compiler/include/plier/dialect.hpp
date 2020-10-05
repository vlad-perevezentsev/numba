#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Function.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "plier/PlierOpsEnums.h.inc"

namespace plier
{
using namespace mlir; // TODO: remove
#include "plier/PlierOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "plier/PlierOps.h.inc"
}

namespace plier
{

namespace detail
{
struct PyTypeStorage;
}

namespace types
{
enum Kind
{
    // Dialect types.
    PyType = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_3_TYPE,
};
}

class PyType : public mlir::Type::TypeBase<plier::PyType, mlir::Type,
                                           detail::PyTypeStorage>
{
public:
    using Base::Base;

    static bool kindof(unsigned kind) { return kind == types::PyType; }

    static PyType get(mlir::MLIRContext *context, mlir::StringRef name);
    static PyType getUndefined(mlir::MLIRContext *context);

    mlir::StringRef getName() const;
};


}
