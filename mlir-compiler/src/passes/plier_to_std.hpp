#pragma once

#include <memory>

namespace mlir
{
class Pass;
}

std::unique_ptr<mlir::Pass> createPlierToStdPass();
