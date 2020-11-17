#pragma once

class PipelineRegistry;

namespace llvm
{
class StringRef;
}

namespace mlir
{
class MLIRContext;
class TypeConverter;
}

void populate_std_type_converter(mlir::MLIRContext& context, mlir::TypeConverter& converter);

void register_plier_to_std_pipeline(PipelineRegistry& registry);

llvm::StringRef plier_to_std_pipeline_name();
