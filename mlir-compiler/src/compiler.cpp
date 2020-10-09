#include "compiler.hpp"

#include <mlir/IR/Module.h>
#include <mlir/IR/Function.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>

#include "utils.hpp"

#include "passes/plier_to_std.hpp"
#include "passes/lower_to_llvm.hpp"

class CompilerContext::CompilerContextImpl
{
public:
    CompilerContextImpl(mlir::MLIRContext& ctx,
                        const CompilerContext::Settings& settings):
        pm(&ctx, settings.verify)
    {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(createPlierToStdPass());

        populate_lower_to_llvm_pipeline(pm);

        if (settings.pass_statistics)
        {
            pm.enableStatistics();
        }
        if (settings.pass_timings)
        {
            pm.enableTiming();
        }
        if (settings.ir_printing)
        {
            ctx.enableMultithreading(false);
            pm.enableIRPrinting();
        }
    }

    void run(mlir::ModuleOp& module)
    {
        if (mlir::failed(pm.run(module)))
        {
            report_error("Compiler pipeline failed");
        }
    }
private:
    mlir::PassManager pm;
};

CompilerContext::CompilerContext(mlir::MLIRContext& ctx, const Settings& settings):
    impl(std::make_unique<CompilerContextImpl>(ctx, settings))
{

}

CompilerContext::~CompilerContext()
{

}

void CompilerContext::run(mlir::ModuleOp module)
{
    impl->run(module);
}
