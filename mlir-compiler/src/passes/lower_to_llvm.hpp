#pragma once

namespace mlir
{
class PassManager;
}

void populate_lower_to_llvm_pipeline(mlir::PassManager& pm);
