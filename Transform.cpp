#include "Transform.h"

using namespace llvm;

namespace {
unsigned rand(unsigned Min, unsigned Max) {
  return Min + std::rand() % (Max - Min);
}

std::pair<unsigned, unsigned> choose2(unsigned Num) {
  unsigned A = rand(0, Num), B;
  do B = rand(0, Num); while (A == B);
  return {A, B};
}
}

// TODO: don't apply this on constant dimension
std::shared_ptr<LayoutDataType>
FactorTransform::apply(const LayoutDataType &Layout) const {
  std::shared_ptr<LayoutDataType> NewLayout = LayoutDataType::copy(Layout);
  // noop if there's nothing to factor
  if (NewLayout->getDims().empty())
    return NewLayout;

  NewLayout->factorInnermost(Factor);
  if (auto *Struct = dyn_cast<LayoutStruct>(NewLayout.get()))
    Struct->reset();
  return NewLayout;
}

std::shared_ptr<LayoutDataType>
SOATransform::apply(const LayoutDataType &Layout) const {
  // noop if the input is not a struct
  if (!isa<LayoutStruct>(&Layout))
    return LayoutDataType::copy(Layout);

  auto &Struct = *cast<LayoutStruct>(&Layout);
  auto NewStruct = std::make_shared<LayoutStruct>(Struct);

  auto Dims = Struct.getDims();
  // noop if input layout is not an array
  if (Dims.empty())
    return LayoutDataType::copy(Layout);

  auto Dim = Dims[0];
  NewStruct->removeInnermostDim();

  for (auto &Field : NewStruct->getFields())
    Field.second->appendDim(Dim);

  NewStruct->reset();

  return NewStruct;
}

std::shared_ptr<LayoutDataType>
AOSTransform::apply(const LayoutDataType &Layout) const {
  if (!isa<LayoutStruct>(&Layout))
    return LayoutDataType::copy(Layout);

  auto &Struct = *cast<LayoutStruct>(&Layout);
  auto NewStruct = std::make_shared<LayoutStruct>(Struct);

  auto Fields = NewStruct->getFields();

  // check if there's a common factor among the outermost dimenion
  // of all the fields
  //
  // TODO: use a more sophisticated technique other than checking if
  // the size expression have the same address
  //
  SymExpr *CommonDim = nullptr;
  for (auto Field : Fields) {
    SymExpr *Dim = Field.second->getDims().rbegin()->first.get();
    if (!CommonDim)
      CommonDim = Dim;
    else if (CommonDim != Dim)
      // fields don't have common outer dimensions, give up
      return NewStruct;
  }

  auto Dim = *Fields[0].second->getDims().rbegin();
  NewStruct->prependDim(Dim);
  for (auto Field : Fields) {
    Field.second->removeOutermostDim();
    if (auto *S = dyn_cast<LayoutStruct>(Field.second.get()))
      S->reset();
  }

  NewStruct->reset();

  return NewStruct;
}

std::shared_ptr<LayoutDataType>
StructTransform::apply(const LayoutDataType &Layout) const {
  if (!isa<LayoutStruct>(&Layout))
    return LayoutDataType::copy(Layout);

  auto &Struct = *cast<LayoutStruct>(&Layout);
  auto NewStruct = std::make_shared<LayoutStruct>(Struct);

  // merging two fields or less doesn't change any thing
  if (Struct.getFields().size() <= 2)
    return NewStruct;

  auto Fields = NewStruct->getFields();
  unsigned Begin = rand(0, Fields.size() - 1),
           End = rand(Begin + 1, Fields.size());
  NewStruct->mergeFields(Begin, End);
  NewStruct->reset();
  return NewStruct;
}

std::shared_ptr<LayoutDataType>
StructFlattenTransform::apply(const LayoutDataType &Layout) const {
  if (!isa<LayoutStruct>(&Layout))
    return LayoutDataType::copy(Layout);

  auto &Struct = *cast<LayoutStruct>(&Layout);
  auto NewStruct = std::make_shared<LayoutStruct>(Struct);

  NewStruct->flatten(rand(0, NewStruct->getFields().size()));
  NewStruct->reset();
  return NewStruct;
}

std::shared_ptr<LayoutDataType>
InterchangeTransform::apply(const LayoutDataType &Layout) const {
  auto NewLayout = LayoutDataType::copy(Layout);
  auto Dims = NewLayout->getDims();

  // not enough dimensions to interchange
  if (Dims.size() < 2)
    return NewLayout;

  unsigned A, B;
  std::tie(A, B) = choose2(Dims.size());
  NewLayout->swapDims(A, B);

  return NewLayout;
}

std::shared_ptr<LayoutDataType>
SwapTransform::apply(const LayoutDataType &Layout) const {
  auto NewLayout = LayoutDataType::copy(Layout);
  auto *Struct = dyn_cast<LayoutStruct>(NewLayout.get());
  if (!Struct)
    return NewLayout;

  auto Fields = Struct->getFields();
  if (Fields.size() < 2)
    return NewLayout;

  unsigned A, B;
  std::tie(A, B) = choose2(Fields.size());

  Struct->swapFields(A, B);
  Struct->reset();

  return NewLayout;
}
