#include "pass_registry.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/StringSaver.h>

#include "utils.hpp"

#include <set>
#include <unordered_map>
#include <utility>

void pass_registry::register_pipeline(pass_registry::registry_entry_t func)
{
    assert(nullptr != func);
    pipelines.push_back(std::move(func));
}

namespace
{
template<typename T, typename IterF, typename VisitF>
void topo_visit(T& elem, IterF&& iter_func, VisitF&& func)
{
    if (elem.visited)
    {
        return;
    }
    elem.visited = true;
    iter_func(elem, [&](T& next)
    {
        topo_visit(next, std::forward<IterF>(iter_func), std::forward<VisitF>(func));
    });
    func(elem);
}
}

void pass_registry::populate_pass_manager(mlir::OpPassManager& pm) const
{
    llvm::BumpPtrAllocator allocator;
    llvm::UniqueStringSaver string_set(allocator);

    using name_id = const void*;
    auto get_id = [](llvm::StringRef name)->name_id
    {
        assert(!name.empty());
        return name.data();
    };
    std::set<llvm::StringRef> pipelines_ordered; // sorted map to make order consistent

    auto get_pipeline = [&](llvm::StringRef name)->llvm::StringRef
    {
        if (name.empty())
        {
            report_error("Empty pipeline name");
        }
        auto str = string_set.save(name);
        pipelines_ordered.insert(str);
        return str;
    };

    struct IdSet : protected llvm::SmallVector<name_id, 4>
    {
        using Base = llvm::SmallVector<name_id, 4>;
        using Base::begin;
        using Base::end;
        void push_back(name_id id)
        {
            auto it = std::equal_range(begin(), end(), id);
            if (it.first == it.second)
            {
                insert(it.first, id);
            }
        }
    };

    struct PipelineInfo
    {
        llvm::StringRef name;
        llvm::SmallVector<llvm::StringRef, 4> prev_pipelines;
        llvm::SmallVector<llvm::StringRef, 4> next_pipelines;
        pipeline_funt_t func = nullptr;
        PipelineInfo* next = nullptr;
        bool visited = false;
    };

    std::unordered_map<name_id, PipelineInfo> pipelines_map;

    auto sink = [&](llvm::StringRef pipeline_name,
                    llvm::ArrayRef<llvm::StringRef> prev_pipelines,
                    llvm::ArrayRef<llvm::StringRef> next_pipelines,
                    pipeline_funt_t func)
    {
        assert(nullptr != func);
        auto i = get_pipeline(pipeline_name);
        auto it = pipelines_map.insert({get_id(i), {}});
        if (!it.second)
        {
            report_error("Duplicated pipeline name");
        }
        auto& info = it.first->second;
        info.name = i;
        info.func = func;
        llvm::transform(prev_pipelines, std::back_inserter(info.prev_pipelines), get_pipeline);
        llvm::transform(next_pipelines, std::back_inserter(info.next_pipelines), get_pipeline);
    };

    for (auto& p : pipelines)
    {
        assert(nullptr != p);
        p(sink);
    }

    auto get_pipeline_info = [&](llvm::StringRef name)->PipelineInfo&
    {
        auto id = get_id(name);
        auto it = pipelines_map.find(id);
        if (it == pipelines_map.end())
        {
            report_error(llvm::Twine("Pipeline not found") + name);
        }
        return it->second;
    };

    // Make all deps bidirectional
    for (auto name : pipelines_ordered)
    {
        auto& info = get_pipeline_info(name);
        for (auto prev : info.prev_pipelines)
        {
            auto& prev_info = get_pipeline_info(prev);
            prev_info.next_pipelines.push_back(name);
        }
        for (auto next : info.next_pipelines)
        {
            auto& next_info = get_pipeline_info(next);
            next_info.prev_pipelines.push_back(name);
        }
    }

    // toposort
    PipelineInfo* first_pipeline = nullptr;
    for (auto name : pipelines_ordered)
    {
        auto iter_func = [&](const PipelineInfo& elem, auto func)
        {
            for (auto prev : elem.prev_pipelines)
            {
                if (get_id(prev) == get_id(name))
                {
                    report_error(llvm::Twine("Pipeline depends on itself: ") + name);
                }
                func(get_pipeline_info(prev));
            }
        };
        auto visit_func = [&](PipelineInfo& elem)
        {
            assert(nullptr == elem.next);
            elem.next = first_pipeline;
            first_pipeline = &elem;
        };
        topo_visit(get_pipeline_info(name), iter_func, visit_func);
    }

    for (auto current = first_pipeline; nullptr != first_pipeline;
         first_pipeline = first_pipeline->next)
     {
        assert(nullptr != current);
        current->func(pm);
     }
}
