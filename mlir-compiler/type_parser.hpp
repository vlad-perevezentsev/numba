#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

mlir::LLVM::LLVMType parse_type(mlir::LLVM::LLVMDialect& dialect, llvm::StringRef str);
