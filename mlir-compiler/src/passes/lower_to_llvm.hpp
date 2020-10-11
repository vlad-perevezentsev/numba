#pragma once

namespace mlir
{
class OpPassManager;
}

void populate_lower_to_llvm_pipeline(mlir::OpPassManager& pm);
