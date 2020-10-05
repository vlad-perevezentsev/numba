#include "compiler.hpp"

#include <mlir/IR/Module.h>
#include <mlir/IR/Function.h>
#include <mlir/Pass/PassManager.h>

#include "utils.hpp"

class CompilerContext::CompilerContextImpl
{
public:
    CompilerContextImpl(mlir::MLIRContext& ctx):
        pm(&ctx)
    {
        auto& funcPm = pm.nest<mlir::FuncOp>();
        // TODO
    }

    mlir::PassManager& get_pm() { return pm; }
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
    if (mlir::failed(impl->get_pm().run(module)))
    {
        report_error("Compiler pipeline failed");
    }
}
