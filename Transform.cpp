#include "Transform.h"
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

namespace {

void resetEverything(LayoutDataType *Layout) {
  if (auto *Struct = dyn_cast<LayoutStruct>(Layout)) {
    for (auto &F : Struct->getFields())
      resetEverything(F.second.get());
    Struct->reset();
  }
}

unsigned rand(unsigned Min, unsigned Max) {
  return Min + std::rand() % (Max - Min);
}

double randProb() {
  static std::default_random_engine Ngn;
  static std::uniform_real_distribution<double> Dist(0.0, 1.0);
  return Dist(Ngn);
}

std::pair<unsigned, unsigned> choose2(unsigned Num) {
  unsigned A = rand(0, Num), B;
  do
    B = rand(0, Num);
  while (A == B);
  return {A, B};
}

} // end anonymous namespace

LayoutDataType *
LayoutTransform::select(LayoutDataType *Layout,
                        std::vector<LayoutDataType *> &Parents) const {
  //  auto *Struct = dyn_cast<LayoutStruct>(Layout);
  //
  //  // got nothing to select from
  //  if (!Struct) {
  //    return Layout;
  //  }
  //
  //
  // double p = randProb();
  // if (p < TransformParams::PRoot) {
  //  return Layout;
  //}

  // auto Fields = Struct->getFields();
  // auto *SubLayout = Fields[rand(0, Fields.size())].second.get();
  // Parents.push_back(Layout);
  // return select(SubLayout, Parents);
  std::vector<LayoutDataType *> Nodes;
  std::vector<LayoutDataType *> Worklist = {Layout};
  while (!Worklist.empty()) {
    LayoutDataType *Layout = Worklist.back();
    Worklist.pop_back();
    auto *Struct = dyn_cast<LayoutStruct>(Layout);
    if (Struct) {
      for (auto &F : Struct->getFields())
        Worklist.push_back(F.second.get());
      Nodes.push_back(Layout);
      assert(isa<LayoutStruct>(Layout) && "WTF");
    } else if (WorksOnScalar)
      Nodes.push_back(Layout);
  }
  return Nodes[rand(0, Nodes.size())];
}

std::unique_ptr<LayoutDataType>
LayoutTransform::apply(const LayoutDataType &Layout) const {
  auto NewLayout = LayoutDataType::copy(Layout);
  // randomly select a sub-layout and apply the transformation
  std::vector<LayoutDataType *> Parents;
  auto *SubLayout = select(NewLayout.get(), Parents);
  transform(SubLayout);
  resetEverything(NewLayout.get());
  return NewLayout;
}

void TransformPool::addTransform(std::unique_ptr<LayoutTransform> Transform,
                                 float Prob) {
  assert(Prob <= 1 && "Probability is greater than 1");

  if (Probs.empty()) {
    Probs.push_back(Prob);
    Transforms.push_back(std::move(Transform));
    return;
  }

  float Acc = Probs.back() + Prob;
  assert(Acc <= 1 && "Sum of inidividual probability greater than 1");

  Probs.push_back(Acc);
  Transforms.push_back(std::move(Transform));
}

std::unique_ptr<LayoutDataType>
TransformPool::apply(const LayoutDataType &Layout) const {
  double p = randProb();
  for (unsigned i = 0, e = Probs.size(); i != e; i++)
    if (Probs[i] > p)
      return Transforms[i]->apply(Layout);

  // apply none of the transformations
  return LayoutDataType::copy(Layout);
}

void FactorTransform::transform(LayoutDataType *Layout) const {
  if (Layout->getDims().empty())
    return;

  // don't factor constant dimension
  if (isa<SymConst>(Layout->getDims()[0].first.get()))
    return;

  Layout->factorInnermost(Factor);
  if (auto *Struct = dyn_cast<LayoutStruct>(Layout))
    Struct->reset();
}

void SOATransform::transform(LayoutDataType *Layout) const {
  auto *Struct = dyn_cast<LayoutStruct>(Layout);

  // noop if the input is not a struct
  if (!Struct)
    return;

  auto Dims = Struct->getDims();
  // noop if input layout is not an array
  if (Dims.empty())
    return;

  auto Dim = Dims[0];
  Struct->removeInnermostDim();
  for (auto &Field : Struct->getFields())
    Field.second->appendDim(Dim);

  Struct->reset();
}

void AOSTransform::transform(LayoutDataType *Layout) const {
  auto *Struct = dyn_cast<LayoutStruct>(Layout);

  if (!Struct)
    return;

  auto Fields = Struct->getFields();

  // check if there's a common factor among the outermost dimenion
  // of all the fields
  //
  // TODO: use a more sophisticated technique other than checking if
  // the size expression have the same address
  //
  SymExpr *CommonDim = nullptr;
  for (auto &Field : Fields) {
    if (Field.second->getDims().empty())
      return;
    SymExpr *Dim = Field.second->getDims().rbegin()->first.get();
    if (!CommonDim)
      CommonDim = Dim;
    else if (CommonDim != Dim)
      // fields don't have common outer dimensions, give up
      return;
  }

  auto Dim = *Fields[0].second->getDims().rbegin();
  Struct->prependDim(Dim);
  for (auto &Field : Fields) {
    Field.second->removeOutermostDim();
    if (auto *S = dyn_cast<LayoutStruct>(Field.second.get()))
      S->reset();
  }

  Struct->reset();
}

void StructTransform::transform(LayoutDataType *Layout) const {
  auto *Struct = dyn_cast<LayoutStruct>(Layout);

  if (!Struct || Struct->getFields().size() <= 2)
    return;

  auto Fields = Struct->getFields();
  unsigned Begin = rand(0, Fields.size() - 1),
           End = rand(Begin + 1, Fields.size());
  Struct->mergeFields(Begin, End);
  Struct->reset();
}

void StructFlattenTransform::transform(LayoutDataType *Layout) const {
  auto *Struct = dyn_cast<LayoutStruct>(Layout);
  if (!Struct)
    return;

  unsigned NumFields = Struct->getFields().size();
  Struct->flatten(rand(0, NumFields));
  Struct->reset();
}

void InterchangeTransform::transform(LayoutDataType *Layout) const {
  auto Dims = Layout->getDims();

  // not enough dimensions to interchange
  if (Dims.size() < 2)
    return;

  unsigned A, B;
  std::tie(A, B) = choose2(Dims.size());
  Layout->swapDims(A, B);
}

void SwapTransform::transform(LayoutDataType *Layout) const {
  auto *Struct = dyn_cast<LayoutStruct>(Layout);
  if (!Struct)
    return;

  auto Fields = Struct->getFields();
  if (Fields.size() < 2)
    return;

  unsigned A, B;
  std::tie(A, B) = choose2(Fields.size());

  Struct->swapFields(A, B);
  Struct->reset();
}
