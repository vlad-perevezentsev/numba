#include "transforms/pipeline_utils.hpp"

#include <mlir/IR/Module.h>
#include <mlir/IR/Attributes.h>

mlir::ModuleOp get_module(mlir::Operation* op)
{
    assert(nullptr != op);
    while (!mlir::isa<mlir::ModuleOp>(op))
    {
        op = op->getParentOp();
        assert(nullptr != op);
    }
    return mlir::cast<mlir::ModuleOp>(op);
}

namespace
{
const constexpr llvm::StringLiteral jump_marker_name("pipeline_jump_markers");
}

mlir::ArrayAttr get_pipeline_jump_markers(mlir::ModuleOp module)
{
    return module.getAttrOfType<mlir::ArrayAttr>(jump_marker_name);
}

void add_pipeline_jump_marker(mlir::ModuleOp module, mlir::StringAttr name)
{
    assert(name);
    assert(!name.getValue().empty());

    llvm::SmallVector<mlir::Attribute, 16> name_list;
    if (auto old_attr = module.getAttrOfType<mlir::ArrayAttr>(jump_marker_name))
    {
        name_list.assign(old_attr.begin(), old_attr.end());
    }
    auto it = llvm::lower_bound(name_list, name,
    [](mlir::Attribute lhs, mlir::StringAttr rhs)
    {
        return lhs.cast<mlir::StringAttr>().getValue() < rhs.getValue();
    });
    if (it == name_list.end())
    {
        name_list.emplace_back(name);
    }
    else if (*it != name)
    {
        name_list.insert(it, name);
    }
    module.setAttr(jump_marker_name, mlir::ArrayAttr::get(name_list, module.getContext()));
}


void remove_pipeline_jump_marker(mlir::ModuleOp module, mlir::StringAttr name)
{
    assert(name);
    assert(!name.getValue().empty());

    llvm::SmallVector<mlir::Attribute, 16> name_list;
    if (auto old_attr = module.getAttrOfType<mlir::ArrayAttr>(jump_marker_name))
    {
        name_list.assign(old_attr.begin(), old_attr.end());
    }
    auto it = llvm::lower_bound(name_list, name,
    [](mlir::Attribute lhs, mlir::StringAttr rhs)
    {
        return lhs.cast<mlir::StringAttr>().getValue() < rhs.getValue();
    });
    assert(it != name_list.end());
    name_list.erase(it);
    module.setAttr(jump_marker_name, mlir::ArrayAttr::get(name_list, module.getContext()));
}
