#pragma once

#include <llvm/ADT/StringRef.h>

class PipelineRegistry;

void register_base_pipeline(PipelineRegistry& registry);

struct PipelineStage
{
    llvm::StringRef begin;
    llvm::StringRef end;
};

PipelineStage get_high_lowering_stage(); // TODO: better name
PipelineStage get_lower_lowering_stage(); // TODO: better name
