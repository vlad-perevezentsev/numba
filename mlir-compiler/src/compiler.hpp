#pragma once

#include <memory>

namespace mlir
{
class MLIRContext;
class ModuleOp;
}

class CompilerContext
{
public:
    class CompilerContextImpl;

    CompilerContext(mlir::MLIRContext& ctx);
    ~CompilerContext();

    CompilerContext(CompilerContext&&) = default;

    void run(mlir::ModuleOp module);

private:
    std::unique_ptr<CompilerContextImpl> impl;
};

void run_compiler(CompilerContext& context, mlir::ModuleOp module);
