#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Function.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "proto/PlierOpsEnums.h.inc"

namespace plier
{
using namespace mlir; // TODO: remove
#include "proto/PlierOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "proto/PlierOps.h.inc"
}

namespace plier
{

void register_dialect();

namespace types
{
enum Kind
{
    // Dialect types.
//    PyState = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_3_TYPE,
//    PyObject,
};
}

}
