#pragma once

class PipelineRegistry;

namespace llvm
{
class StringRef;
}

void register_plier_to_linalg_pipeline(PipelineRegistry& registry);

llvm::StringRef plier_to_linalg_pipeline_name();
