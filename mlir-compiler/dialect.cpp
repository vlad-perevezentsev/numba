#include "plier/dialect.hpp"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Builders.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace plier
{

void register_dialect()
{
    mlir::registerDialect<mlir::StandardOpsDialect>();
    mlir::registerDialect<plier::PlierDialect>();
}

PlierDialect::PlierDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
    addOperations<
#define GET_OP_LIST
#include "plier/PlierOps.cpp.inc"
        >();
    addTypes<plier::PyType>();
}

mlir::Type PlierDialect::parseType(mlir::DialectAsmParser &parser) const {
    parser.emitError(parser.getNameLoc(), "unknown type");
    return mlir::Type();
}

void PlierDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &os) const {
    switch (type.getKind()) {
    case plier::types::PyType:
        os << "PyType";
        return;
    default:
        llvm_unreachable("unexpected type kind");
    }
}

void ArgOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  unsigned index, mlir::StringRef name) {
    ArgOp::build(builder, state, PyType::get(state.getContext()), llvm::APInt(32, index), name);
}

void CastOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value val) {
    CastOp::build(builder, state, PyType::get(state.getContext()), val);
}

void ConstOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Attribute val) {
    ConstOp::build(builder, state, PyType::get(state.getContext()), val);
}

#define GET_OP_CLASSES
#include "plier/PlierOps.cpp.inc"

}
#include "plier/PlierOpsEnums.cpp.inc"
