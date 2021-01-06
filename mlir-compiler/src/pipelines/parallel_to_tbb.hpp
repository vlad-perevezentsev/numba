#pragma once

class PipelineRegistry;

namespace llvm
{
class StringRef;
}

void register_parallel_to_tbb_pipeline(PipelineRegistry& registry);

llvm::StringRef parallel_to_tbb_pipeline_name();
