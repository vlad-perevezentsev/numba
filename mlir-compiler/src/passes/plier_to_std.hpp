#pragma once

namespace mlir
{
class OpPassManager;
}

void populate_plier_to_std_pipeline(mlir::OpPassManager& pm);
