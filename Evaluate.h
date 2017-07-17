#ifndef EVALUATE_H
#define EVALUATE_H

#include <vector>

namespace llvm {
  class Module;
}

// assume M is contains the bitcode for a whole program
// compile it, run it with Args, and return time it takes to run
double evaluate(llvm::Module &M, const std::vector<std::string> &Args);

#endif
