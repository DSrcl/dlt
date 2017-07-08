#include <map>

namespace llvm {
  class Value;
  class Constant;
  class StructType;
  class Instruction;
  class DSNode;
}

class IndexTriple;

class TransformLowering {
protected:
  llvm::StructType *TargetTy;
  llvm::Constant *ElemSize;
  std::map<unsigned, unsigned> Offset2IdxMap;

  TransformLowering(const llvm::DSNode *Target);

public:
  llvm::Constant *getOrigSize() const { return ElemSize; }

  // compute the address of an element after transformation
  virtual llvm::Value *ComputeAddress(IndexTriple Triple, unsigned Offset,
                                Instruction *InsertPt) = 0;

  // calculate number of bytes needed for the object after the transformation
  // `Size` is number of elements allocated (not bytes) in the original
  // allocation
  virtual Value *ComputeBytesToAlloc(Value *Size, Instruction *InsertPt) = 0;
};

// a class to test the correctness pointer-to-triple conversion
class NoopTransform : public TransformLowering {
public:
  NoopTransform(const DSNode *Target) : TransformLowering(Target) {}
  Value *ComputeAddress(IndexTriple Triple, unsigned Offset,
                        Instruction *InsertPt) override;
  Value *ComputeBytesToAlloc(Value *Size, Instruction *InsertPt) override {
    return BinaryOperator::CreateNSWMul(Size, ElemSize, "NewSize", InsertPt);
  }
};

//
// This lowers a layout to IR by implementing the address and size calculation
// of a an object after layout transformation
//
class LayoutLowering : public TransformLowering {
  const DataLayout *TD;
  //
  // for each field, f in the the original layout
  // mapping f -> the path from which f can be reached after transformation,
  // (if one were to view a data type as a tree)
  //
  std::map<unsigned, std::vector<const LayoutDataType *>> Paths;

  std::string SizeVar, IdxVar;

  std::shared_ptr<LayoutDataType> Layout;

public:
  LayoutLowering(const DSNode *Target,
                 std::shared_ptr<LayoutDataType> TheLayout, std::string SizeVar,
                 std::string IdxVar, const DataLayout *TD);

  // ============================== interface ==============================
  Value *ComputeAddress(IndexTriple Triple, unsigned Offset,
                        Instruction *InsertPt) override;
  Value *ComputeBytesToAlloc(Value *Size, Instruction *InsertPt) override;
};

