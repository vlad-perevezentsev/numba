#include "passes/base_pipeline.hpp"

#include "pipeline_registry.hpp"

void register_base_pipeline(PipelineRegistry& registry)
{
    auto dummu_func = [](mlir::OpPassManager&){};
    registry.register_pipeline([&](auto sink)
    {
        sink("init", {}, {}, dummu_func);
    });
    registry.register_pipeline([&](auto sink)
    {
        sink("lowering", {"init"}, {}, dummu_func);
    });
    registry.register_pipeline([&](auto sink)
    {
        sink("terminate", {"lowering"}, {}, dummu_func);
    });
}
