#include "pass_registry.hpp"


void pass_registry::register_pipeline(pass_registry::registry_entry_t func)
{
    assert(nullptr != func);
    pipelines.push_back(std::move(func));
}

void pass_registry::populate_pass_manager(mlir::OpPassManager& pm) const
{
    // TODO: build proper dep graph
    auto sink = [&](llvm::StringRef /*pipeline_name*/,
                    llvm::ArrayRef<llvm::StringRef> /*prev_pipelines*/,
                    llvm::ArrayRef<llvm::StringRef> /*next_pipelines*/,
                    pipeline_funt_t func)
    {
        assert(nullptr != func);
        func(pm);
    };
    for (auto& p : pipelines)
    {
        assert(nullptr != p);
        p(sink);
    }
}
