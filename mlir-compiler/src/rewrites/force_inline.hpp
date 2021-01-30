#pragma once

#include <mlir/IR/PatternMatch.h>

namespace mlir
{
class CallOp;
}

struct ForceInline : public mlir::OpRewritePattern<mlir::CallOp>
{
    using mlir::OpRewritePattern<mlir::CallOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::CallOp op, mlir::PatternRewriter &rewriter) const override;
};
