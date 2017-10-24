#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <vector>
#include "Layout.h"

namespace TransformParams {
  // probability of choosing a root struct instead of descending
  // while selecting sub-layout
  const float PRoot = 0.5;
}

//
// a mapping from layout -> layout
//
class LayoutTransform {
  // randomly walk down the layout to select a sublayout and remember nodes preceding the node selected
  LayoutDataType *select(LayoutDataType *Layout, std::vector<LayoutDataType *> &Parents) const;
protected:
  bool WorksOnScalar;
public:
  LayoutTransform(bool WorksOnScalar_) : WorksOnScalar(WorksOnScalar_) {}
  virtual void transform(LayoutDataType *) const = 0;
  std::unique_ptr<LayoutDataType> apply(const LayoutDataType &) const;
};

//
// randomly select one from a pool of transformations and apply it
//
class TransformPool {
  std::vector<float> Probs;
  std::vector<std::unique_ptr<LayoutTransform>> Transforms;
public:
  // add a transformation to the pool, Prob is the probability with which we apply it
  void addTransform(std::unique_ptr<LayoutTransform> Transform, float Prob); 
  std::unique_ptr<LayoutDataType> apply(const LayoutDataType &) const;
};

//
// Factor
//  Split the innermost dimension of a layout into two dimensions
//  This is just a wrapper around LayoutDataType::factorInnermost to keep
//  the transform interface consistent
//
class FactorTransform : public LayoutTransform {
  ExprPtr Factor;

public:
  FactorTransform(unsigned TheFactor)
    : LayoutTransform(true), Factor(Const(TheFactor)) {}
  void transform(LayoutDataType *) const override;
};

//
// AOS-to-SOA
//  peel of the inner most dimension and append to each elements
//
class SOATransform : public LayoutTransform {
public:
  SOATransform() : LayoutTransform(false) {}
  void transform(LayoutDataType *) const override;
};

//
// SOA-to-AOS
//  inverse of AOS-to-SOA. If the outermost dimensions of all fields in a struct
//  are identical, pull it out as the innermost of the struct's
//
class AOSTransform : public LayoutTransform {
public:
  AOSTransform() : LayoutTransform(false) {}
  void transform(LayoutDataType *) const override;
};

//
// *randomly* merge consecutive fields within a struct into a single field of
// struct
//
class StructTransform : public LayoutTransform {
public:
  StructTransform() : LayoutTransform(false) {}
  void transform(LayoutDataType *) const override;
};

//
// *randomly* select a field within a struct, if the field is also a struct,
// remove the struct and replace it with its fields. This is essentially
// a wrapper around LayoutStruct::flatten
//
class StructFlattenTransform : public LayoutTransform {
public:
  StructFlattenTransform() : LayoutTransform(false) {}
  void transform(LayoutDataType *) const override;
};

//
// *Randomly* swaps two dimensions
//
class InterchangeTransform : public LayoutTransform {
public:
  InterchangeTransform() : LayoutTransform(true) {}
  void transform(LayoutDataType *) const override;
};

//
// If the layout is a struct, *randomly* swap two fields
//
class SwapTransform : public LayoutTransform {
public:
  SwapTransform() : LayoutTransform(false) {}
  void transform(LayoutDataType *) const override;
};

#endif
