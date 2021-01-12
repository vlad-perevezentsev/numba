#include "utils.hpp"

#include <stdexcept>

#include "llvm/ADT/Twine.h"

void report_error(const llvm::Twine& msg)
{
    throw std::runtime_error(msg.str());
}
