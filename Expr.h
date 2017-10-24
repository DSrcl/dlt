#ifndef EXPR_H
#define EXPR_H

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>
#include <map>
#include <memory>

namespace llvm {
  class Value;
  class Type;
}

//
// symexpr = constant
//          | variable
//          | symexpr % symexpr
//          | symexpr / symexpr
//          | symexpr * symexpr
//          | symexpr & symexpr
//          | symexpr + symexpr
//
class SymExpr {
public:
  enum ExprKind { EK_Const, EK_Var, EK_Mod, EK_Div, EK_Mul, EK_Add, EK_And };

private:
  ExprKind Kind;

protected:
public:
  virtual void dump() const = 0;
  ExprKind getKind() const { return Kind; }
  SymExpr(ExprKind TheKind) : Kind(TheKind) {}
  // lower this expression to LLVM IR
  virtual llvm::Value *emitCode(const std::map<std::string, llvm::Value *> &Vars,
                          llvm::Instruction *InsertPt) = 0;
};

typedef std::shared_ptr<SymExpr> ExprPtr;

struct SymConst : public SymExpr {
  unsigned Val;
  SymConst(unsigned V) : SymExpr(EK_Const), Val(V) {}
  static bool classof(const SymExpr *SE) { return SE->getKind() == EK_Const; }
  llvm::Value *emitCode(const std::map<std::string, llvm::Value *> &Vars,
                  llvm::Instruction *InsertPt) override {
    using namespace llvm;
    auto &Ctx = InsertPt->getContext();
    return ConstantInt::get(Type::getInt64Ty(Ctx), Val);
  }
  void dump() const override {
    llvm::errs() << Val;
  }
};

struct SymVar : public SymExpr {
  std::string Name;
  SymVar(std::string TheName) : SymExpr(EK_Var), Name(TheName) {}
  static bool classof(const SymExpr *SE) { return SE->getKind() == EK_Var; }
  llvm::Value *emitCode(const std::map<std::string, llvm::Value *> &Vars,
                  llvm::Instruction *InsertPt) override {
    assert(Vars.count(Name) && "undefined variable");
    return Vars.at(Name);
  }
  void dump() const override {
    llvm::errs() << "var(" << Name << ")";
  }
};

struct SymMod : public SymExpr {
  ExprPtr Left, Right;
  SymMod(ExprPtr L, ExprPtr R)
      : SymExpr(EK_Mod), Left(std::move(L)), Right(std::move(R)) {}
  static bool classof(const SymExpr *SE) { return SE->getKind() == EK_Mod; }
  llvm::Value *emitCode(const std::map<std::string, llvm::Value *> &Vars,
                  llvm::Instruction *InsertPt) override {
    using namespace llvm;
    Value *L = Left->emitCode(Vars, InsertPt),
          *R = Right->emitCode(Vars, InsertPt);
    return BinaryOperator::CreateURem(L, R, "", InsertPt);
  }
  void dump() const override {
    Left->dump();
    llvm::errs() << "%";
    Right->dump();
  }
};

struct SymDiv : public SymExpr {
  ExprPtr Left, Right;
  bool RoundUp;
  SymDiv(ExprPtr L, ExprPtr R, bool RoundUp_)
      : SymExpr(EK_Div), Left(std::move(L)), Right(std::move(R)),
        RoundUp(RoundUp_) {}
  static bool classof(const SymExpr *SE) { return SE->getKind() == EK_Div; }
  llvm::Value *emitCode(const std::map<std::string, llvm::Value *> &Vars,
                  llvm::Instruction *InsertPt) override {
    using namespace llvm;
    Value *L = Left->emitCode(Vars, InsertPt),
          *R = Right->emitCode(Vars, InsertPt);
    if (RoundUp) {
      auto &Ctx = InsertPt->getContext();
      auto *One = ConstantInt::get(Type::getInt64Ty(Ctx), 1);
      auto *RMinus1 = BinaryOperator::CreateNSWSub(R, One, "", InsertPt);
      L = BinaryOperator::CreateNSWAdd(L, RMinus1, "", InsertPt);
    }
    return BinaryOperator::CreateUDiv(L, R, "", InsertPt);
  }
  void dump() const override {
    Left->dump();
    llvm::errs() << "/";
    Right->dump();
  }
};

struct SymMul : public SymExpr {
  ExprPtr Left, Right;
  SymMul(ExprPtr L, ExprPtr R)
      : SymExpr(EK_Mul), Left(std::move(L)), Right(std::move(R)) {}
  static bool classof(const SymExpr *SE) { return SE->getKind() == EK_Mul; }
  llvm::Value *emitCode(const std::map<std::string, llvm::Value *> &Vars,
                  llvm::Instruction *InsertPt) override {
    using namespace llvm;
    Value *L = Left->emitCode(Vars, InsertPt),
          *R = Right->emitCode(Vars, InsertPt);
    return BinaryOperator::CreateNSWMul(L, R, "", InsertPt);
  }
  void dump() const override {
    Left->dump();
    llvm::errs() << "*";
    Right->dump();
  }
};

struct SymAdd : public SymExpr {
  ExprPtr Left, Right;
  SymAdd(ExprPtr L, ExprPtr R)
      : SymExpr(EK_Add), Left(std::move(L)), Right(std::move(R)) {}
  static bool classof(const SymExpr *SE) { return SE->getKind() == EK_Add; }
  llvm::Value *emitCode(const std::map<std::string, llvm::Value *> &Vars,
                  llvm::Instruction *InsertPt) override {
    using namespace llvm;
    Value *L = Left->emitCode(Vars, InsertPt),
          *R = Right->emitCode(Vars, InsertPt);
    return BinaryOperator::CreateNSWAdd(L, R, "", InsertPt);
  }
  void dump() const override {
    llvm::errs() << "(";
    Left->dump();
    llvm::errs() << "+";
    Right->dump();
    llvm::errs() << ")";
  }
};

struct SymAnd : public SymExpr {
  ExprPtr Left, Right;
  SymAnd(ExprPtr L, ExprPtr R)
      : SymExpr(EK_And), Left(std::move(L)), Right(std::move(R)) {}
  static bool classof(const SymExpr *SE) { return SE->getKind() == EK_And; }
  llvm::Value *emitCode(const std::map<std::string, llvm::Value *> &Vars,
                  llvm::Instruction *InsertPt) override {
    using namespace llvm;
    Value *L = Left->emitCode(Vars, InsertPt),
          *R = Right->emitCode(Vars, InsertPt);
    return BinaryOperator::CreateAnd(L, R, "", InsertPt);
  }
  void dump() const override {
    Left->dump();
    llvm::errs() << "&";
    Right->dump();
  }
};

inline ExprPtr operator*(ExprPtr A, ExprPtr B) {
  return std::make_shared<SymMul>(A, B);
}

inline ExprPtr operator+(ExprPtr A, ExprPtr B) {
  return std::make_shared<SymAdd>(A, B);
}

inline ExprPtr Const(unsigned V) { return std::make_shared<SymConst>(V); }

#endif
