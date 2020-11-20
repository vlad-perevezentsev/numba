#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>

#include <functional>
#include <vector>

namespace mlir
{
class OpPassManager;
}


class PipelineRegistry
{
public:
    PipelineRegistry() = default;
    PipelineRegistry(const PipelineRegistry&) = delete;

    using pipeline_funt_t = void(*)(mlir::OpPassManager&);
    using registry_entry_sink_t = void(
        llvm::StringRef pipeline_name,
        llvm::ArrayRef<llvm::StringRef> prev_pipelines,
        llvm::ArrayRef<llvm::StringRef> next_pipelines,
        llvm::ArrayRef<llvm::StringRef> jumps,
        pipeline_funt_t func);
    using registry_entry_t = std::function<void(llvm::function_ref<registry_entry_sink_t>)>;

    void register_pipeline(registry_entry_t func);

    void populate_pass_manager(mlir::OpPassManager& pm) const;

private:
    std::vector<registry_entry_t> pipelines;
};
