#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "Evaluate.h"

using namespace llvm;

namespace {
  ExitOnError ExitOnErr;
}

double evaluate(Module &M, const std::vector<std::string> &Args) {
  std::string Clang = ExitOnErr(errorOrToExpected(sys::findProgramByName("clang++")));
  SmallVector<char, 64> Bitcode, Exe;
  ExitOnErr(errorCodeToError(sys::fs::createTemporaryFile("", "bc", Bitcode)));
  ExitOnErr(errorCodeToError(sys::fs::createTemporaryFile("", "", Exe)));
  FileRemover X(Bitcode), Y(Exe);

  {
    // write the module to disk
    std::error_code EC;
    tool_output_file BcOut(Bitcode.data(), EC, sys::fs::F_None);
    ExitOnErr(errorCodeToError(EC));

    legacy::PassManager Passes;
    Passes.add(createBitcodeWriterPass(BcOut.os(), true));
    Passes.add(createVerifierPass());
    Passes.run(M);
    BcOut.keep();
  }
  
  const char *CompileArgs[] = { "clang++", Bitcode.data(), "-O3", "-o", Exe.data(), nullptr };
  if (sys::ExecuteAndWait(Clang, CompileArgs)) {
    exit(1);
  }

  const char *ExecArgs[Args.size()+2];
  int i = 0;
  ExecArgs[i++] = Exe.data();
  for (auto &Arg : Args)
    ExecArgs[i++] = Arg.data();
  ExecArgs[i] = nullptr;
  
  Timer ExecTimer("", "");
  ExecTimer.startTimer();
  StringRef Empty = "";
  const StringRef *Redirects[] = { &Empty, &Empty, &Empty };
  if (sys::ExecuteAndWait(Exe.data(), ExecArgs, nullptr/*env*/, Redirects)) {
    errs() << "Failed to execute compile program\n";
    exit(1);
  }
  ExecTimer.stopTimer();

  return ExecTimer.getTotalTime().getWallTime();
}
