#include "passes/plier_to_linalg.hpp"

#include "plier/dialect.hpp"

#include "passes/plier_to_std.hpp"

#include "base_pipeline.hpp"
#include "pipeline_registry.hpp"

namespace
{

void populate_plier_to_linalg_pipeline(mlir::OpPassManager& /*pm*/)
{

}
}

void register_plier_to_linalg_pipeline(PipelineRegistry& registry)
{
    registry.register_pipeline([](auto sink)
    {
        auto stage = get_high_lowering_stage();
        sink(plier_to_linalg_pipeline_name(), {plier_to_std_pipeline_name()}, {stage.end}, &populate_plier_to_linalg_pipeline);
    });
}

llvm::StringRef plier_to_linalg_pipeline_name()
{
    return "plier_to_linalg";
}
