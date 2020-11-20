#pragma once

namespace mlir
{
class ArrayAttr;
class Operation;
class ModuleOp;
class StringAttr;
}

mlir::ModuleOp get_module(mlir::Operation* op);
mlir::ArrayAttr get_pipeline_jump_markers(mlir::ModuleOp module);
void add_pipeline_jump_marker(mlir::ModuleOp module, mlir::StringAttr name);
void remove_pipeline_jump_marker(mlir::ModuleOp module, mlir::StringAttr name);
