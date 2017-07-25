#include "Layout.h"
#include "dsa/DSGraph.h"
#include <memory>

using namespace llvm;

//
// get the number of bytes this type aligned to
// e.g. in x86, 8 for `n x { double, int }`
//
unsigned LayoutDataType::getAlignedSize(const DataLayout *TD) const {
  //
  // single elem
  //
  if (getDims().empty())
    return TD->getABITypeAlignment(getRightmostType());

  //
  // array
  //
  return TD->getABITypeAlignment(getLeftmostType());
}

LayoutStruct::LayoutStruct(const LayoutStruct &Src)
  : LayoutDataType(Src), TD(Src.TD) {
  Fields.resize(Src.Fields.size());
  for (unsigned i = 0, e = Fields.size(); i != e; i++) {
    Fields[i].first = Src.Fields[i].first;
    Fields[i].second = LayoutDataType::copy(*Src.Fields[i].second);
  }
  // TODO: reuse calculations from Src
  reset();
}

void LayoutStruct::reset() {
  const LayoutDataType *FirstField = Fields.begin()->second.get();
  const LayoutDataType *PrevField = FirstField;
  ExprPtr Size = FirstField->getSizeForAll(TD);
  OffsetOf.clear();
  OffsetOf[Fields.begin()->second.get()] = Const(0);

  LLVMType = nullptr;

  bool representableByLLVMType = FirstField->getType() != nullptr;
  std::vector<Type *> FieldTypes { FirstField->getType() };

  for (auto fi = std::next(Fields.begin()), fe = Fields.end(); fi != fe; ++fi) {
    const LayoutDataType *Field = fi->second.get();
    Type *FTy = Field->getType();
    representableByLLVMType &= FTy != nullptr;
    representableByLLVMType &= Field->hasConstDims();
    FieldTypes.push_back(FTy);

    // need to align previous field with current one
    if (PrevField->getAlignedSize(TD) <
        TD->getABITypeAlignment(Field->getLeftmostType()))
      Size = alignTo(Size, Field->getLeftmostType(), TD);

    OffsetOf[Field] = Size;

    Size = Size + Field->getSizeForAll(TD);
    PrevField = Field;
  }

  if (representableByLLVMType) {
    auto &Ctx = FieldTypes[0]->getContext();
    assert(FieldTypes.size() == Fields.size());
    LLVMType = StructType::get(Ctx, FieldTypes);
  } else
    LLVMType = nullptr;

  // if this is an array we also need to align the last element with
  // the first one
  if (!getDims().empty() &&
      Fields.begin()->second->getAlignedSize(TD) <
          Fields.rbegin()->second->getAlignedSize(TD))
    Size = alignTo(Size, getRightmostType(), TD);

  SizeForOne = Size;
}

LayoutStruct::LayoutStruct(std::vector<Field> &TheFields, const DataLayout *TD_)
    : LayoutDataType(DTK_Struct), TD(TD_) {
  Fields.resize(TheFields.size());
  for (unsigned i = 0, e = Fields.size(); i != e; i++) {
    Fields[i].first = TheFields[i].first;
    Fields[i].second = std::move(TheFields[i].second);
  }
  reset();
}

LayoutStruct *LayoutStruct::create(const DSNode *Src, const DataLayout *TD,
                                   std::string SizeVar, std::string IdxVar) {
  std::vector<Field> Fields;
  unsigned i = 0;
  for (auto ti = Src->type_begin(), te = Src->type_end(); ti != te; ++ti, ++i) {
    Type *Ty = const_cast<Type *>(*ti->second->begin());
    Fields.push_back({i, make_unique<LayoutScalar>(Ty)});
  }
  auto *Layout = new LayoutStruct(Fields, TD);
  // assume this object is an array
  Layout->appendDim(
      {std::make_shared<SymVar>(SizeVar), std::make_shared<SymVar>(IdxVar)});
  return Layout;
}

ExprPtr LayoutDataType::getIdxExpr() const {
  if (Dimensions.empty())
    return Const(0);

  ExprPtr Expr = Dimensions[0].second;
  for (unsigned i = 1; i < Dimensions.size(); i++)
    Expr = Expr + Dimensions[i - 1].first * Dimensions[i].second;
  return Expr;
}

void LayoutDataType::factorInnermost(ExprPtr Factor) {
  assert(!Dimensions.empty() && "no dimensions to factor");
  LayoutDimension &Innermost = Dimensions[0];

  ExprPtr N = Innermost.first, // size expression
      I = Innermost.second;    // index expression

  auto FactoredN = std::make_shared<SymDiv>(N, Factor, true /* round up */);
  auto FactoredI = std::make_shared<SymDiv>(I, Factor, false);

  Innermost = {FactoredN, FactoredI};
  prependDim({Factor, std::make_shared<SymMod>(I, Factor)});
}

std::unique_ptr<LayoutDataType>
LayoutDataType::copy(const LayoutDataType &Layout) {
  if (isa<LayoutScalar>(&Layout))
    return make_unique<LayoutScalar>(*cast<LayoutScalar>(&Layout));

  return make_unique<LayoutStruct>(*cast<LayoutStruct>(&Layout));
}

bool LayoutDataType::hasConstDims() const {
  auto isConstDim = [](const LayoutDimension &Dim) {
    SymExpr *DimSize = Dim.first.get();
    return isa<SymConst>(DimSize);
  };

  return all_of(Dimensions.begin(), Dimensions.end(), isConstDim);
}

Type *LayoutDataType::getType() const {
  Type *ETy = getElementType();
  if (!ETy) return nullptr;

  Type *CurTy = ETy;
  for (auto &Dim : Dimensions) {
    SymExpr *SizeExpr = Dim.first.get();
    if (auto *Size = dyn_cast<SymConst>(SizeExpr)) {
      CurTy = ArrayType::get(CurTy, Size->Val);
    } else
      // give up if one of the dimenion is not const
      return nullptr;
  }

  return CurTy;
}

void LayoutStruct::mergeFields(unsigned Begin, unsigned End) {
  auto BeginIt = Begin > 0 ? std::next(Fields.begin(), Begin) : Fields.begin(),
       EndIt = std::next(Fields.begin(), End + 1);
  std::vector<LayoutStruct::Field> FieldsToMerge(
      std::make_move_iterator(BeginIt),
      std::make_move_iterator(EndIt));
  auto MergedField = make_unique<LayoutStruct>(FieldsToMerge, TD);
  Fields.erase(std::next(BeginIt), EndIt);
  *BeginIt = std::make_pair(-1, std::move(MergedField));
}

void LayoutStruct::flatten(unsigned i) {
  assert(i < Fields.size());
  auto *FieldToRemove = Fields[i].second.get();
  if (!FieldToRemove->getDims().empty())
    return;

  auto *InnerStruct = dyn_cast<LayoutStruct>(FieldToRemove);
  if (!InnerStruct)
    return;

  Fields.insert(Fields.begin() + i + 1,
      std::make_move_iterator(InnerStruct->Fields.begin()),
      std::make_move_iterator(InnerStruct->Fields.end()));
  Fields.erase(Fields.begin() + i);
}
