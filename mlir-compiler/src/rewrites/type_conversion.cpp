#include "rewrites/type_conversion.hpp"

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include "plier/dialect.hpp"

namespace
{
mlir::LogicalResult setBlockSig(
    mlir::Block& block, mlir::OpBuilder& builder,
    const mlir::TypeConverter::SignatureConversion& conversion)
{
    if (conversion.getConvertedTypes().size() != block.getNumArguments())
    {
        return mlir::failure();
    }
    unsigned i = 0;
    for (auto it : llvm::zip(block.getArguments(), conversion.getConvertedTypes()))
    {
        auto arg = std::get<0>(it);
        auto type = std::get<1>(it);
        if (arg.getType() != type)
        {
            builder.setInsertionPointToStart(&block);
            auto res = builder.create<plier::CastOp>(builder.getUnknownLoc(), arg.getType(), arg);
            arg.replaceUsesWithIf(res, [&](mlir::OpOperand& op)
            {
                return op.getOwner() != res;
            });

            for (auto& use : block.getUses())
            {
                auto op = use.getOwner();
                builder.setInsertionPoint(op);
                if (auto br = mlir::dyn_cast<mlir::BranchOp>(op))
                {
                    assert(&block == br.dest());
                    auto src = br.destOperands()[i];
                    auto new_op = builder.create<plier::CastOp>(op->getLoc(), type, src);
                    br.destOperandsMutable().slice(i, 1).assign(new_op);
                }
                else if (auto cond_br = mlir::dyn_cast<mlir::CondBranchOp>(op))
                {
                    if (&block == cond_br.trueDest())
                    {
                        auto src = cond_br.trueDestOperands()[i];
                        auto new_op = builder.create<plier::CastOp>(op->getLoc(), type, src);
                        cond_br.trueDestOperandsMutable().slice(i, 1).assign(new_op);
                    }
                    if (&block == cond_br.falseDest())
                    {
                        auto src = cond_br.falseDestOperands()[i];
                        auto new_op = builder.create<plier::CastOp>(op->getLoc(), type, src);
                        cond_br.falseDestOperandsMutable().slice(i, 1).assign(new_op);
                    }
                }
                else
                {
                    llvm_unreachable("setBlockSig: unknown operation type");
                }
            }
            arg.setType(type);
        }
        ++i;
    }
    return mlir::success();
}

mlir::LogicalResult convertRegionTypes(
    mlir::Region *region, mlir::TypeConverter &converter, bool apply)
{
    assert(nullptr != region);
    if (region->empty())
    {
        return mlir::failure();
    }

    mlir::OpBuilder builder(region->getContext());

    // Convert the arguments of each block within the region.
    auto sig = converter.convertBlockSignature(&region->front());
    assert(static_cast<bool>(sig));
    if (apply)
    {
        auto res = setBlockSig(region->front(), builder, *sig);
        assert(mlir::succeeded(res));
        (void)res;
    }
    for (auto &block : llvm::make_early_inc_range(llvm::drop_begin(*region, 1)))
    {
        sig = converter.convertBlockSignature(&block);
        if (!sig)
        {
            return mlir::failure();
        }
        if (apply)
        {
            if (mlir::failed(setBlockSig(block, builder, *sig)))
            {
                return mlir::failure();
            }
        }
    }
    return mlir::success();
}
}

FuncOpSignatureConversion::FuncOpSignatureConversion(mlir::TypeConverter& conv,
    mlir::MLIRContext* ctx)
    : OpRewritePattern(ctx), converter(conv) {}

mlir::LogicalResult FuncOpSignatureConversion::matchAndRewrite(
    mlir::FuncOp funcOp, mlir::PatternRewriter& rewriter) const
{
    auto type = funcOp.getType();

    // Convert the original function types.
    mlir::TypeConverter::SignatureConversion result(type.getNumInputs());
    llvm::SmallVector<mlir::Type, 1> newResults;
    if (mlir::failed(converter.convertSignatureArgs(type.getInputs(), result)) ||
        mlir::failed(converter.convertTypes(type.getResults(), newResults)) ||
        mlir::failed(convertRegionTypes(&funcOp.getBody(), converter, false)))
    {
        return mlir::failure();
    }

    // Update the function signature in-place.
    rewriter.updateRootInPlace(funcOp, [&] {
        funcOp.setType(mlir::FunctionType::get(result.getConvertedTypes(), newResults,
                                               funcOp.getContext()));
        auto res = convertRegionTypes(&funcOp.getBody(), converter, true);
        assert(mlir::succeeded(res));
    });
    return mlir::success();
}
