#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Function.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

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

void register_dialect();

namespace types
{
enum Kind
{
    // Dialect types.
    PyType = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_3_TYPE,
};
}

class PyType : public mlir::Type::TypeBase<PyType, mlir::Type, mlir::TypeStorage> {
public:
    using Base::Base;
    static bool kindof(unsigned kind) { return kind == types::PyType; }
    static PyType get(mlir::MLIRContext *context)
    {
        return Base::get(context, types::PyType);
    }
};


}
