#pragma once

namespace llvm
{
class Twine;
}

[[noreturn]] void report_error(const llvm::Twine& msg);
