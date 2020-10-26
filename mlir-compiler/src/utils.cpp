#include "utils.hpp"

#include <stdexcept>

#include "llvm/ADT/Twine.h"

void report_error(const llvm::Twine& msg)
{
    auto str = msg.str();
    throw std::exception(str.c_str());
}
