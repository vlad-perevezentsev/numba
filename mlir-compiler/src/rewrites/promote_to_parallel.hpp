#pragma once

#include <mlir/IR/PatternMatch.h>

namespace mlir
{
namespace scf
{
class ForOp;
}
}

struct PromoteToParallel : public mlir::OpRewritePattern<mlir::scf::ForOp>
{
    using mlir::OpRewritePattern<mlir::scf::ForOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::scf::ForOp op, mlir::PatternRewriter &rewriter) const override;
};
