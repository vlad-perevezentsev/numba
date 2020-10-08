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
    CompilerContextImpl(mlir::MLIRContext& ctx):
        pm(&ctx, /*verify*/true)
    {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(createPlierToStdPass());

        populate_lower_to_llvm_pipeline(pm);

        pm.enableStatistics();
        pm.enableTiming();
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

CompilerContext::CompilerContext(mlir::MLIRContext& ctx):
    impl(std::make_unique<CompilerContextImpl>(ctx))
{

}

CompilerContext::~CompilerContext()
{

}

void CompilerContext::run(mlir::ModuleOp module)
{
    impl->run(module);
}
