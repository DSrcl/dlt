#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "Layout.h"

//
// interface that provides a mapping from layout -> layout
//
// TODO: this currently only applies to the *root* node of a layout tree
//
class LayoutTransform {
public:
  virtual std::shared_ptr<LayoutDataType>
  apply(const LayoutDataType &) const = 0;
};

//
// Factor
//  Split the innermost dimension of a layout into two dimensions
//  This is just a wrapper around LayoutDataType::factorInnermost to keep
//  the transform interface consistent
//
class FactorTransform : public LayoutTransform {
  // factor with which we factor (my english is so good it makes me cringe)
  ExprPtr Factor;

public:
  FactorTransform(unsigned TheFactor) : Factor(Const(TheFactor)) {}
  std::shared_ptr<LayoutDataType> apply(const LayoutDataType &) const override;
};

//
// AOS-to-SOA
//  peel of the inner most dimension and append to each elements
//
class SOATransform : public LayoutTransform {
public:
  std::shared_ptr<LayoutDataType> apply(const LayoutDataType &) const override;
};

//
// SOA-to-AOS
//  inverse of AOS-to-SOA. If the outermost dimensions of all fields in a struct
//  are identical, pull it out as the innermost of the struct's
//
class AOSTransform : public LayoutTransform {
public:
  std::shared_ptr<LayoutDataType> apply(const LayoutDataType &) const override;
};

//
// *randomly* merge consecutive fields within a struct into a single field of
// struct
//
class StructTransform : public LayoutTransform {
public:
  std::shared_ptr<LayoutDataType> apply(const LayoutDataType &) const override;
};

//
// *randomly* select a field within a struct, if the field is also a struct,
// remove the struct and replace it with its fields. This is essentially
// a wrapper around LayoutStruct::flatten
//
class StructFlattenTransform : public LayoutTransform {
public:
  std::shared_ptr<LayoutDataType> apply(const LayoutDataType &) const override;
};

//
// *Randomly* swaps two dimensions
//
class InterchangeTransform : public LayoutTransform {
public:
  std::shared_ptr<LayoutDataType> apply(const LayoutDataType &) const override;
};

//
// If the layout is a struct, *randomly* swap two fields
//
class SwapTransform : public LayoutTransform {
public:
  std::shared_ptr<LayoutDataType> apply(const LayoutDataType &) const override;
};

#endif
