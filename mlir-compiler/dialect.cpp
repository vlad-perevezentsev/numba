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
    ArgOp::build(builder, state, PyType::get(state.getContext()),
                 llvm::APInt(32, index), name);
}

void ConstOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,

                   mlir::Attribute val) {
    ConstOp::build(builder, state, PyType::get(state.getContext()), val);
}

void GlobalOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     mlir::StringRef name) {
    GlobalOp::build(builder, state, PyType::get(state.getContext()), name);
}

void BinOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs, mlir::StringRef op) {
    BinOp::build(builder, state, PyType::get(state.getContext()), lhs, rhs, op);
}

void CastOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value val) {
    CastOp::build(builder, state, PyType::get(state.getContext()), val);
}

void PyCallOp::build(OpBuilder &builder, OperationState &state, mlir::Value func,
                     mlir::ValueRange args,
                     mlir::ArrayRef<std::pair<std::string, mlir::Value>> kwargs) {
    auto ctx = builder.getContext();
    mlir::SmallVector<mlir::Value, 16> all_args;
    all_args.reserve(args.size() + kwargs.size());
    std::copy(args.begin(), args.end(), std::back_inserter(all_args));
    auto kw_start = llvm::APInt(32, all_args.size());
    mlir::SmallVector<mlir::Attribute, 8> kw_names;
    kw_names.reserve(kwargs.size());
    for (auto& a : kwargs)
    {
        kw_names.push_back(mlir::StringAttr::get(a.first, ctx));
        all_args.push_back(a.second);
    }
    PyCallOp::build(builder, state, PyType::get(state.getContext()), func,
                    all_args, kw_start, mlir::ArrayAttr::get(kw_names, ctx));
}

#define GET_OP_CLASSES
#include "plier/PlierOps.cpp.inc"

}
#include "plier/PlierOpsEnums.cpp.inc"
