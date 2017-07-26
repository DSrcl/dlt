#include "TransformState.h"
#include "dsa/DSGraph.h"

using namespace llvm;

static double randProb() {
  return std::rand() / double(RAND_MAX);
}

void LayoutState::mutate() {
  if (!Cur)
    Prev = Orig;
  else
    Prev = Cur;
  
  Cur = TP.apply(*getLayout());
}

void LayoutState::revert() {
  if (Prev)
    Cur = Prev;
}

void LayoutState::reset() {
  Prev = Cur;
  Cur = Orig;
}

const LayoutDataType *LayoutState::getLayout() const {
  if (!Cur)
    return Orig.get();
  return Cur.get();
}

TransformState::TransformState(LayoutSet &OrigLayouts, const TransformPool &TP, double MutateProb, double ResetProb) 
  : MutateP(MutateProb), ResetP(ResetProb), BestSoFar(-1) {
  Mutated.resize(OrigLayouts.size());
  for (auto &IdAndLayout : OrigLayouts)
    States.emplace_back(
        IdAndLayout.first, LayoutState(std::unique_ptr<const LayoutDataType>(IdAndLayout.second), TP));
}

void TransformState::mutate() {
  if (randProb() < ResetP) {
    reset();
    return;
  }

  for (unsigned i = 0, e = States.size(); i != e; i++)
    if (randProb() < MutateP) {
      States[i].second.mutate();
      Mutated[i] = true;
    } else
      Mutated[i] = false;
}

void TransformState::setCost(double Cost) {
  if (BestSoFar == -1 || Cost < BestSoFar) {
    BestSoFar = Cost;
    Best.clear();
    for (auto &IdAndState : States)
      Best.emplace_back(
          IdAndState.first,
          LayoutDataType::copy(*IdAndState.second.getLayout()));
  }
}

void TransformState::revert() {
  for (unsigned i = 0, e = States.size(); i != e; i++)
    if (Mutated[i])
      States[i].second.revert();
}

void TransformState::reset() {
  for (auto &IdAndState : States)
    IdAndState.second.reset();
}

LayoutSet TransformState::getLayouts() {
  LayoutSet Layouts;
  for (auto &IdAndState : States)
    Layouts.emplace_back(IdAndState.first, IdAndState.second.getLayout());
  return Layouts;
}

LayoutSet TransformState::getBest() {
  LayoutSet Layouts;
  for (auto &IdAndLayout: Best)
    Layouts.emplace_back(IdAndLayout.first, IdAndLayout.second.get());
  return Layouts;
}
