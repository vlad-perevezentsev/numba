#pragma once

class PipelineRegistry;

namespace llvm
{
class StringRef;
}

void register_lower_to_llvm_pipeline(PipelineRegistry& registry);

llvm::StringRef lower_to_llvm_pipeline_name();
