#include "plier_to_std.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace
{
struct PlierToStdPass :
    public mlir::PassWrapper<PlierToStdPass, mlir::FunctionPass>
{
    void runOnFunction()
    {

    }
};
}

std::unique_ptr<mlir::Pass> createPlierToStdPass()
{
    return std::make_unique<PlierToStdPass>();
}
