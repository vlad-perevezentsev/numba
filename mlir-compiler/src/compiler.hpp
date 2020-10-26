#pragma once

#include <memory>

namespace mlir
{
class MLIRContext;
class ModuleOp;
}

class PipelineRegistry;
class CompilerContext
{
public:
    struct Settings
    {
        bool verify = false;
        bool pass_statistics = false;
        bool pass_timings = false;
        bool ir_printing = false;
    };

    class CompilerContextImpl;

    CompilerContext(mlir::MLIRContext& ctx, const Settings& settings,
                    const PipelineRegistry& registry);
    ~CompilerContext();

    CompilerContext(CompilerContext&&) = default;

    void run(mlir::ModuleOp module);

private:
    std::unique_ptr<CompilerContextImpl> impl;
};

void run_compiler(CompilerContext& context, mlir::ModuleOp module);
