#include "compiler.hpp"

#include <mlir/IR/Module.h>
#include <mlir/IR/Function.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <mlir/IR/Diagnostics.h>

#include <llvm/Support/raw_ostream.h>

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
        populate_plier_to_std_pipeline(pm);
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
        std::string err;
        llvm::raw_string_ostream err_stream(err);
        auto diag_handler = [&](mlir::Diagnostic& diag)
        {
            if (diag.getSeverity() == mlir::DiagnosticSeverity::Error)
            {
                err_stream << diag;
            }
        };

        scoped_diag_handler(*pm.getContext(), diag_handler, [&]()
        {
            if (mlir::failed(pm.run(module)))
            {
                err_stream.flush();
                report_error(llvm::Twine("MLIR pipeline failed\n") + err);
            }
        });
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
