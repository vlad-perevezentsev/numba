#pragma once

#include <mlir/IR/Function.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir
{
class TypeConverter;
}

struct FuncOpSignatureConversion : public mlir::OpRewritePattern<mlir::FuncOp>
{
    FuncOpSignatureConversion(mlir::MLIRContext* ctx,
                              mlir::TypeConverter& conv);

    /// Hook for derived classes to implement combined matching and rewriting.
    mlir::LogicalResult
    matchAndRewrite(mlir::FuncOp funcOp, mlir::PatternRewriter &rewriter) const override;

private:
    mlir::TypeConverter& converter;
};

struct OpTypeConversion : public mlir::RewritePattern
{
    OpTypeConversion(mlir::MLIRContext* ctx,
                     mlir::TypeConverter& conv);

    /// Hook for derived classes to implement combined matching and rewriting.
    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter &rewriter) const override;

private:
    mlir::TypeConverter& converter;
};
