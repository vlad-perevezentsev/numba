#include "passes/plier_to_linalg.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include "plier/dialect.hpp"

#include "passes/plier_to_std.hpp"

#include "base_pipeline.hpp"
#include "pipeline_registry.hpp"

namespace
{

struct PlierToLinalgPass :
    public mlir::PassWrapper<PlierToLinalgPass, mlir::OperationPass<mlir::ModuleOp>>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<plier::PlierDialect>();
        registry.insert<mlir::StandardOpsDialect>();
    }

    void runOnOperation() override;
};

void PlierToLinalgPass::runOnOperation()
{
//    mlir::TypeConverter type_converter;
//    type_converter.addConversion([](plier::Type type)->llvm::Optional<mlir::Type>
//    {
//        return map_plier_type(type);
//    });

    mlir::OwningRewritePatternList patterns;
//    patterns.insert<FuncOpSignatureConversion,
//                    OpTypeConversion>(&getContext(), type_converter);
//    patterns.insert<ConstOpLowering, BinOpLowering,
//                    CallOpLowering, CastOpLowering,
//                    ExpandTuples>(&getContext());

    auto apply_conv = [&]()
    {
        (void)mlir::applyPatternsAndFoldGreedily(getOperation(), patterns);
    };

    apply_conv();
}

void populate_plier_to_linalg_pipeline(mlir::OpPassManager& pm)
{
    pm.addPass(std::make_unique<PlierToLinalgPass>());
}
}

void register_plier_to_linalg_pipeline(PipelineRegistry& registry)
{
    registry.register_pipeline([](auto sink)
    {
        auto stage = get_high_lowering_stage();
        sink(plier_to_linalg_pipeline_name(), {plier_to_std_pipeline_name()}, {stage.end}, &populate_plier_to_linalg_pipeline);
    });
}

llvm::StringRef plier_to_linalg_pipeline_name()
{
    return "plier_to_linalg";
}
