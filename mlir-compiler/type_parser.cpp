#include "type_parser.hpp"

#include <mlir/Parser.h>

namespace
{
[[noreturn]] void report_error(const llvm::Twine& msg)
{
    auto str = msg.str();
    throw std::exception(str.c_str());
}
}

mlir::LLVM::LLVMType parse_type(mlir::LLVM::LLVMDialect& dialect, llvm::StringRef str)
{
    assert(!str.empty());
    auto mlir_type = (std::string("!llvm<\"") + str + "\">").str();
    auto res = mlir::parseType(mlir_type, dialect.getContext()).dyn_cast_or_null<mlir::LLVM::LLVMType>();
    if (mlir::Type() == res)
    {
        report_error(llvm::Twine("cannot parse type: \"") + str + "\"");
    }
    return res;
}
