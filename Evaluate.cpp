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
cl::list<std::string> ExtraObjs("extra-obj", cl::desc("<extra objs to build the test program with>"));
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
  
  const char *CompileFlags[] = { "-O3", "-o", Exe.data(),
    // TODO: use cl::list instead of hardcoding these
    "-lz", "-lpthread", "-lrt", "-lutil",
    "-ldl", "-lgmp",
    "-lboost_program_options", "-lminisat",
    "-lexpat", "-lfreetype", "-lfontconfig", "-lz", "-licuuc", "-lpng", "-lwebp", "-lwebpdemux", "-lwebpmux",
    "-lXpm", "-lSM", "-lICE", "-lX11", "-ltiff", "-ljpeg", "-lboost_thread" , "-lboost_system",
    "-lutil",
    nullptr };
  std::vector<const char *> CompileArgs { "clang++" };
  for (auto &Obj : ExtraObjs)
    CompileArgs.push_back(Obj.data());
  CompileArgs.push_back(Bitcode.data());
  for (const char *Flag : CompileFlags)
    CompileArgs.push_back(Flag);
  if (sys::ExecuteAndWait(Clang, static_cast<const char**>(CompileArgs.data()))) {
    return -1;
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
  if (int RetCode = sys::ExecuteAndWait(Exe.data(), ExecArgs, nullptr/*env*/, Redirects)) {
    errs() << "Failed to execute compile program, return code = " << RetCode <<"\n";
    return -1;
  }
  ExecTimer.stopTimer();

  return ExecTimer.getTotalTime().getWallTime();
}
