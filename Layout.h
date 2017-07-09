#ifndef LAYOUT_H
#define LAYOUT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/DataLayout.h"
#include "Expr.h"

namespace llvm {
  class DSNode;
}

//
// We represent a dimension as a pair of symbolic expression,
// specifically, the first expression repsents size of the dimension,
// and the second one represents the expression we use for indexing
//
typedef std::pair<ExprPtr, ExprPtr> LayoutDimension;

class LayoutDataType {
  //
  // Dimensions are ordered from innermost to outermost
  //
  std::vector<LayoutDimension> Dimensions;

public:
  enum DataTypeKind { DTK_Scalar, DTK_Struct };

private:
  DataTypeKind Kind;

public:
  DataTypeKind getKind() const { return Kind; }
  LayoutDataType(DataTypeKind TheKind) : Kind(TheKind) {}
  void appendDim(LayoutDimension Dim) { Dimensions.push_back(Dim); }

  void prependDim(LayoutDimension Dim) {
    Dimensions.insert(Dimensions.begin(), Dim);
  }

  void removeInnermostDim() {
    assert(!Dimensions.empty());
    Dimensions.erase(Dimensions.begin());
  }

  void removeOutermostDim() {
    assert(!Dimensions.empty());
    Dimensions.pop_back();
  }

  llvm::ArrayRef<LayoutDimension> getDims() const { return Dimensions; }

  virtual llvm::Type *getLeftmostType() const = 0;
  virtual llvm::Type *getRightmostType() const = 0;
  virtual ExprPtr getSizeForOne(const llvm::DataLayout *) const = 0;
  unsigned getAlignedSize(const llvm::DataLayout *TD) const;

  //
  // size of the whole layout
  //
  ExprPtr getSizeForAll(const llvm::DataLayout *TD) const {
    ExprPtr Size = getSizeForOne(TD);
    for (auto Dim : getDims())
      Size = Size * Dim.first;
    return Size;
  }

  //
  // return expression that computes the next integer that's greater or equal
  // to Value and is aligned with `TypeToAlign`
  //
  static ExprPtr alignTo(ExprPtr Value, llvm::Type *TypeToAlign,
                         const llvm::DataLayout *TD) {
    auto Alignment = Const(TD->getABITypeAlignment(TypeToAlign));
    auto AlignmentMinusOne = Const(TD->getABITypeAlignment(TypeToAlign) - 1);
    return std::make_shared<SymDiv>(Value + AlignmentMinusOne, Alignment,
                                    false /*round up*/) *
           Alignment;
  }

  //
  // get the (linearized) expression used to index into this data
  //
  ExprPtr getIdxExpr() const;

  // factor the innermost dimension into two dimension
  void factorInnermost(ExprPtr Factor);

  static std::unique_ptr<LayoutDataType> copy(const LayoutDataType &Layout);

  //
  // Return the llvm type for this layout
  // `{ n x a, n x b }` -- nullptr (not equivalent)
  // `n x { a, b }` -- { a, b }
  // `4 x { a, b }` -- { a, b }
  //
  virtual llvm::Type *getElementType() const = 0;

  //
  // Similar to getElementType except taking dimensions into account
  // `4 x { a, b }` -- `4 x { a, b }`
  // `n x int` -- int
  //
  llvm::Type *getType() const;

  bool hasConstDims() const;

  void swapDims(unsigned A, unsigned B) {
    assert(A < Dimensions.size() && B < Dimensions.size() && "invalid dimensions index");
    std::swap(Dimensions[A], Dimensions[B]);
  }
};

class LayoutScalar : public LayoutDataType {
  llvm::Type *Ty;

public:
  llvm::Type *getElementType() const override { return Ty; }

  LayoutScalar(llvm::Type *TheType) : LayoutDataType(DTK_Scalar), Ty(TheType) {}
  static bool classof(const LayoutDataType *DT) {
    return DT->getKind() == DTK_Scalar;
  }

  llvm::Type *getLeftmostType() const override { return Ty; }

  llvm::Type *getRightmostType() const override { return Ty; }

  ExprPtr getSizeForOne(const llvm::DataLayout *TD) const override {
    return Const(TD->getTypeAllocSize(Ty));
  }
};

class LayoutStruct : public LayoutDataType {
  //
  // Each field is tagged with the original idx before transformation.
  // Being tagged with -1 means the field is a result of transformation
  // and doesn't correspond to any original field
  //
public:
  typedef std::pair<int, std::unique_ptr<LayoutDataType>> Field;

private:
  const llvm::DataLayout *TD;

  std::vector<Field> Fields;

  ExprPtr SizeForOne;

  // mapping a field to its offset from beginning of the struct
  std::map<const LayoutDataType *, ExprPtr> OffsetOf;

  llvm::Type *LLVMType = nullptr;

public:
  //
  // 1) compute size of the struct
  // 2) compute offset of each field within this struct
  // 3) compute the llvm type for this layout if able to
  //
  // Need to be called after the struct is mutated.
  //
  void reset();

  llvm::ArrayRef<Field> getFields() const { return Fields; }

  ExprPtr getOffsetOf(const LayoutDataType *Field) const {
    assert(OffsetOf.count(Field) && "field is not in current struct");
    return OffsetOf.at(Field);
  }

  LayoutStruct(std::vector<Field> &TheFields, const llvm::DataLayout *TD);
  LayoutStruct(const LayoutStruct &);
  static LayoutStruct *create(const llvm::DSNode *Src, const llvm::DataLayout *TD,
                              std::string SizeVar, std::string IdxVar);

  //
  // merge the fields indexed from Begin to End into a single field of struct.
  // End is inclusive here.
  //
  void mergeFields(unsigned Begin, unsigned End);

  // if the ith field of the current layout is a struct
  // replace that field and insert all the fields in the struct
  // e.g. `flatten {{a,b}, c}, 0` = {a, b, c}
  void flatten(unsigned i);

  static bool classof(const LayoutDataType *DT) {
    return DT->getKind() == DTK_Struct;
  }

  llvm::Type *getLeftmostType() const override {
    return Fields.begin()->second->getLeftmostType();
  }

  llvm::Type *getRightmostType() const override {
    return Fields.rbegin()->second->getRightmostType();
  }

  ExprPtr getSizeForOne(const llvm::DataLayout *TD) const override {
    return SizeForOne;
  }

  void swapFields(unsigned A, unsigned B) {
    assert(A < Fields.size() && B < Fields.size() && "invalid field index");
    std::swap(Fields[A], Fields[B]);
  }

  llvm::Type *getElementType() const override {
    return LLVMType;
  }
};

#endif
