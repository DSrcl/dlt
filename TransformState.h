#ifndef TRANSFORM_STATE_H
#define TRANSFORM_STATE_H

#include "Layout.h"
#include "Transform.h"
#include <memory>
#include <vector>

class LayoutDataType;
namespace llvm {
class DSNode;
}

typedef std::vector<std::pair<unsigned, const LayoutDataType *>> LayoutSet;

//
// helper class used to tracker the state of a single layout
//
class LayoutState {
  std::shared_ptr<const LayoutDataType> Orig, Cur, Prev;
  const TransformPool &TP;

public:
  LayoutState(std::unique_ptr<const LayoutDataType> OrigLayout,
                 const TransformPool &TheTP)
      : Orig(std::move(OrigLayout)), TP(TheTP) {}
  void mutate();
  void revert();
  void reset();
  // get current layout
  const LayoutDataType *getLayout() const;
};

//
// This is a class that tracks layouts of a set of target.
// Here are its primary operations:
//  1) mutate -- transform current layout to a *potentially* different layout
//  2) revert -- go back to previous layout,
//               note that you can't revert twice in a row without 
//               calling `mutate` in between
//  3) reset -- go back to *original* layout of the target
class TransformState {
  std::vector<std::pair<unsigned, LayoutState>> States;
  std::vector<std::pair<unsigned, std::unique_ptr<LayoutDataType>>> Best;
  std::vector<bool> Mutated;
  double MutateP;
  double ResetP;
  double BestSoFar;
public:
  TransformState(LayoutSet &OrigLayouts, const TransformPool &TP,
      double MutateProb, double ResetProb);
  void mutate();
  // remember the performance of current layout
  void setCost(double);
  void revert();
  void reset();
  // get current layout
  LayoutSet getLayouts();
  // get the best layout so far
  LayoutSet getBest();
};

#endif
