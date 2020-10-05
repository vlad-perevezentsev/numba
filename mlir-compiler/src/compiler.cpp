#include "compiler.hpp"

#include <mlir/IR/Module.h>
#include <mlir/IR/Function.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "utils.hpp"

class CompilerContext::CompilerContextImpl
{
public:
    CompilerContextImpl(mlir::MLIRContext& ctx):
        pm(&ctx)
    {
        pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
        auto& funcPm = pm.nest<mlir::FuncOp>();
        // TODO
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
