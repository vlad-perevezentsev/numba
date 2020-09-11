#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

mlir::LLVM::LLVMType parse_type(mlir::MLIRContext& context, llvm::StringRef str);
