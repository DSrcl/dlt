#include "Evaluate.h"
#include "Transform.h"
#include "TransformState.h"
#include "dsa/DSGraph.h"
#include "dsa/DataStructure.h"
#include "dsa/InitializeDSAPasses.h"
#include "dsa/TypeSafety.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <limits>
#include <sys/wait.h>
#include <unistd.h>

#define DEBUG_TYPE "dlt"

//
// ========== RAAAAAAAAAAAAAAAAAAAAAAAAAMBLING ==========
// 0) support use of pthread_create where callee is staically known
// 1) Support use of memcpy that copies constant number of bytes and
// the bytes equals nodes we are transforming
// 2) When getting new type for GEP, relying on the input type is wrong.
// e.g. when `{ char[offset_of_3rd_elem], int }, something, 0, 1 }` is used
// instead of
// `gep { int, int *ptr2target, int }, something, 0, 3`.
// -- Need a helper function that takes old offset and returns offset after
// triple conversion.
// will work. With this, the example above will be translated into
// `gep {char[offset_of_3rd_elem_after_conv], something, 0, 1}`
// <<<< can probably put this off though, considering most program are not
// pathological as this
//
//

using namespace llvm;
using namespace dsa;

namespace llvm {
void initializeLayoutTunerPass(PassRegistry &);
}

namespace {
cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"),
                                   cl::Required);
cl::opt<std::string> OutputFilename("o", cl::desc("Specify output file name"),
                                    cl::value_desc("output file"));
cl::opt<unsigned>
    MinSize("min-size", cl::desc("Minimum size of an object to be transformed"),
            cl::init(4));

cl::opt<unsigned>
    MinFields("min-fields",
              cl::desc("Minimum number of fields an object to be transformed"),
              cl::init(2));

cl::list<std::string>
    TestArgv(cl::ConsumeAfter,
             cl::desc("<program arguments to test compiled module>"));

cl::opt<bool> ArrayOnly("array-only",
                        cl::desc("Only tune objects that are arrays"));

cl::opt<bool> NoPointer("no-pointer",
                        cl::desc("Don't tune objects that contains pointer field(s)"));

cl::opt<bool> NoFP("no-fp",
    cl::desc("Don't tune objects with function pointers"));

cl::opt<double> ResetProb(
    "reset-prob",
    cl::desc(
        "Probility with which a set of layout is reset to original layout"),
    cl::init(0.05));

cl::opt<double>
    MutateProb("mutate-prob",
               cl::desc("Probability with which a single layout is mutated"),
               cl::init(0.2));

cl::opt<unsigned>
    Iterations("iters", cl::desc("Number of iterations to run the tuner with"),
               cl::init(4));

cl::opt<bool> Dump("dump", cl::desc("Dump modules"), cl::init(false));

cl::opt<double> TMax("tmax", cl::desc("Max temperature"), cl::init(0.01));

cl::opt<double> TMin("tmin", cl::desc("Min temperature"), cl::init(0.0001));

double randProb() {
  static std::default_random_engine Ngn;
  static std::uniform_real_distribution<double> Dist(0.0, 1.0);
  return Dist(Ngn);
}

struct IndexTriple {
  Value *Base, *Idx, *Size;
};

typedef std::vector<Value *> GEPIdxTy;
typedef std::map<unsigned, Type *> FieldsTy;

std::string getNewName() {
  static unsigned Counter = 0;
  return "tmp" + std::to_string(Counter++);
}

//
// ===== functions for debugging =====
//

std::unique_ptr<LayoutDataType> toSOA(const LayoutDataType &Layout) {
  return SOATransform().apply(Layout);
}

std::unique_ptr<LayoutDataType> toAOS(const LayoutDataType &Layout) {
  return AOSTransform().apply(Layout);
}

std::unique_ptr<LayoutDataType> factorBy4(const LayoutDataType &Layout) {
  return FactorTransform(4).apply(Layout);
}

std::unique_ptr<LayoutDataType> struct_(const LayoutDataType &Layout) {
  return StructTransform().apply(Layout);
}

std::unique_ptr<LayoutDataType> structFlatten(const LayoutDataType &Layout) {
  return StructFlattenTransform().apply(Layout);
}

std::unique_ptr<LayoutDataType> interchange(const LayoutDataType &Layout) {
  return InterchangeTransform().apply(Layout);
}

std::unique_ptr<LayoutDataType> swapFields(const LayoutDataType &Layout) {
  return SwapTransform().apply(Layout);
}

class TransformLowering {
private:
  const DSNode *Target;

protected:
  StructType *TargetTy;
  Constant *ElemSize;
  unsigned Size;
  std::map<unsigned, unsigned> Offset2IdxMap;

  TransformLowering(const DSNode *TheTarget);

public:
  Constant *getOrigSize() const { return ElemSize; }
  unsigned getSize() const { return Size; }
  const DSNode *getTarget() { return Target; }

  // compute the address of an element after transformation
  virtual Value *ComputeAddress(IndexTriple Triple, unsigned Offset,
                                Instruction *InsertPt,
                                bool DontCare = false) = 0;

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
                        Instruction *InsertPt, bool DontCare) override;
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

  const LayoutDataType *Layout;

public:
  LayoutLowering(const DSNode *Target, const LayoutDataType *TheLayout,
                 std::string SizeVar, std::string IdxVar, const DataLayout *TD);

  // ============================== interface ==============================
  Value *ComputeAddress(IndexTriple Triple, unsigned Offset,
                        Instruction *InsertPt, bool DontCare) override;
  Value *ComputeBytesToAlloc(Value *Size, Instruction *InsertPt) override;
};

// a set of target nodes tagged with its group idx
typedef DenseMap<const DSNode *, unsigned> TargetMapTy;

// mapping a DSNode's offset from a target node
typedef DenseMap<const DSNode *, unsigned> OffsetMapTy;

class CloneSignature {
  std::vector<std::pair<const DSNode *, unsigned>> Targets;
  std::vector<std::pair<const DSNode *, unsigned>> Offsets;
  Function *Src;

public:
  CloneSignature(TargetMapTy TheTargets, OffsetMapTy TheOffsets,
                 Function *TheSrc)
      : Src(TheSrc) {
    for (auto TargetAndGroup : TheTargets)
      Targets.push_back(TargetAndGroup);
    for (auto TargetAndOffset : TheOffsets)
      Offsets.push_back(TargetAndOffset);
  }

  bool operator<(const CloneSignature &Other) const {
    return std::tie(Targets, Offsets, Src) <
           std::tie(Other.Targets, Other.Offsets, Other.Src);
  }
};

class LayoutTuner : public ModulePass {
  typedef std::set<const DSNode *> NodeSetTy;
  // edge between two SCCs
  typedef std::pair<CallSite, const Function *> CallEdgeTy;
  // target datalayout
  const DataLayout *TD;
  DataStructures *DSA;
  AllocIdentify *AllocIdentifier;

  //
  // Helper class to record the fact of one DSNode refering to
  // other DSNodes.
  //
  // Consider this example:
  //============================
  // struct Vec { int size; int *data; };
  //
  // Vec *make_vec() {  return new Vec; }
  //
  // void init_vec(Vec *v, n) { v->data = new int[n]; }
  //
  // void foo(int n) {
  //    Vec *v = make_vec();
  //    init_vec(v, n);
  //    // ... do something with v ...
  // }
  // ============================
  //
  // Say we want to triple-convert the target, v->data, in foo.
  // When we traverse the call graph to make_vec (say we use BU),
  // in the context of make_vec, we don't know if the expression
  // `new Vec` refers to our target since the DSNode of v->data is not
  // even in the DSGraph of make_vec. To fix this, when we begin our
  // traversal from foo, we propagate this information with RefRecord
  // from foo to make_vec.
  class RefRecord {
    const DSNode *OrigN;
    unsigned OffsetToOrig;

    // offsets in a DSNode that could point to a target
    std::set<unsigned> References;

  public:
    // is this node pointing to any target
    bool hasReferences() const { return !References.empty(); }

    bool hasRefAt(unsigned Offset) const { return References.count(Offset); };

    operator bool() const { return hasReferences(); }

    // refers nothing
    RefRecord() {}

    //
    // remember the fact that N points to a subset of nodes in Targets
    //
    template <typename NODE_SET_TY>
    RefRecord(const DSNode *N, const NODE_SET_TY &Targets)
        : OrigN(N), OffsetToOrig(0) {
      for (auto ei = N->edge_begin(), ee = N->edge_end(); ei != ee; ++ei) {
        if (Targets.count(ei->second.getNode()))
          References.insert(ei->first);
      }
    }

    const DSNode *getOrigNode() const { return OrigN; }
    unsigned getOffsetToOrigNode() const { return OffsetToOrig; }

    //
    // remember the fact that CallerNH points to a set of targets
    // in the context of Callee
    //
    RefRecord(DSNodeHandle CallerNH, const DSNode *CalleeN,
              const RefRecord &CallerRefs)
        : OrigN(CallerRefs.OrigN),
          OffsetToOrig(CallerRefs.OffsetToOrig + CallerNH.getOffset()) {
      unsigned Offset = CallerNH.getOffset();
      for (unsigned RefOffset : CallerRefs.References) {
        if (RefOffset < Offset)
          continue;
        References.insert(RefOffset - Offset);
      }
    }
  };

  //
  // Track which nodes refers to targets
  //
  template <typename NODE_SET_TY> class RefSetTracker {
    NODE_SET_TY Targets;
    // mapping DSNode -> set of fields that are references to targets
    std::map<const DSNode *, RefRecord> RefRecords;

  public:
    RefSetTracker() {}
    RefSetTracker(const RefSetTracker &) = default;
    RefSetTracker &operator=(const RefSetTracker &) = default;
    RefSetTracker(RefSetTracker &&) = default;
    RefSetTracker(const NODE_SET_TY &TheTargets) : Targets(TheTargets) {}

    const NODE_SET_TY &getTargets() const { return Targets; }
    NODE_SET_TY &getTargets() { return Targets; }

    //
    // propagate reference info from caller to callee
    //
    RefSetTracker
    getCalleeTracker(const NODE_SET_TY &TargetsInCallee,
                     const DSGraph::NodeMapTy &CalleeCallerMapping) const {
      RefSetTracker CalleeTracker(TargetsInCallee);
      DEBUG(errs() << "SIZE OF TARGETS: " << Targets.size() << '\n');

      const DSNode *CalleeN;
      DSNodeHandle CallerNH;
      // propagate RefRecords
      for (auto Mapping : CalleeCallerMapping) {
        std::tie(CalleeN, CallerNH) = Mapping;
        auto *CallerN = CallerNH.getNode();
        auto RefIt = RefRecords.find(CallerN);
        if (RefIt != RefRecords.end()) {
          const auto &CallerRefRecord = RefIt->second;
          CalleeTracker.RefRecords[CalleeN] =
              RefRecord(CallerNH, CalleeN, CallerRefRecord);
        } else if (auto Ref = RefRecord(CallerN, Targets)) {
          CalleeTracker.RefRecords[CalleeN] = RefRecord(CallerNH, CalleeN, Ref);
          assert(CalleeTracker.RefRecords.at(CalleeN).getOrigNode() ==
                 CallerNH.getNode());
        }
      }

      return CalleeTracker;
    }

    // assuming N points to a target,
    // find the top-level N that correspond to it
    const DSNode *getOrigNode(const DSNode *N) const {
      auto RefIt = RefRecords.find(N);
      if (RefIt != RefRecords.end())
        return RefIt->second.getOrigNode();
      return N;
    }

    unsigned getOffsetToOrigNode(const DSNode *N) const {
      auto RefIt = RefRecords.find(N);
      if (RefIt != RefRecords.end())
        return RefIt->second.getOffsetToOrigNode();
      return 0;
    }

    bool hasRefAt(const DSNode *N, unsigned Offset) const {
      if (Offset < N->getSize() && N->hasLink(Offset) &&
          Targets.count(N->getLink(Offset).getNode()))
        return true;

      auto It = RefRecords.find(N);
      if (It != RefRecords.end())
        return It->second.hasRefAt(Offset);

      return false;
    }

    bool refersTargets(const DSNode *N) const {
      if (!N)
        return false;

      if (RefRecords.count(N))
        return true;

      for (auto ei = N->edge_begin(), ee = N->edge_end(); ei != ee; ++ei)
        if (Targets.count(ei->second.getNode()))
          return true;

      return false;
    }
  };

  // We use this for looking up DSGraph/DSNode after cloning
  struct FunctionCloneRecord {
    Function *Src;
    std::map<const Value *, const Value *> CloneToSrcValueMap;
  };
  std::map<const Function *, FunctionCloneRecord> CloneRecords;

  // we can reuse a clone if they have the same parent and same type
  std::map<CloneSignature, Function *> CloneCache;

  // reallocs for different targets
  std::map<const DSNode *, Function *> Reallocs;

  Type *VoidTy, *Int8Ty, *Int32Ty, *Int64Ty;
  StructType *TripleTy;
  PointerType *Int8PtrTy, *TriplePtrTy;
  Constant *Zero, *One, *Two, *Zero64, *One64;

  TargetMapTy findLegalTargets(const Module &M);

  // mapping value -> value it replaced
  std::map<const Value *, const Value *> ReplacedValueMap;

  // mapping value -> its original type
  std::map<const Value *, Type *> ReplacedTypeMap;

  const Value *getOrigValue(const Value *V) const {
    auto ValIt = ReplacedValueMap.find(V);
    if (ValIt != ReplacedValueMap.end())
      return ValIt->second;
    return V;
  }

  void updateValue(const Value *Old, const Value *NewV) {
    ReplacedValueMap[NewV] = getOrigValue(Old);
  }

  void updateType(Value *V, Type *NewTy) {
    Type *OldTy = getOrigType(V);
    V->mutateType(NewTy);
    ReplacedTypeMap[getOrigValue(V)] = OldTy;
  }

  Type *getOrigType(Value *V) {
    auto *OrigVal = getOrigValue(V);
    auto It = ReplacedTypeMap.find(OrigVal);
    if (It != ReplacedTypeMap.end())
      return It->second;
    return OrigVal->getType();
  }

  DSNodeHandle getNodeForValue(const Value *V, const Function *F) const {
    // F might be a clone
    auto CloneIt = CloneRecords.find(F);
    if (CloneIt != CloneRecords.end()) {
      auto &CR = CloneIt->second;
      F = CR.Src;
      auto ValIt = CR.CloneToSrcValueMap.find(getOrigValue(V));
      if (ValIt != CR.CloneToSrcValueMap.end())
        V = ValIt->second;
      else
        return DSNodeHandle();
    }

    return DSA->getDSGraph(*F)->getNodeForValue(getOrigValue(V));
  }

  Function *getSrcFunc(const Function *F) const {
    if (CloneRecords.find(F) != CloneRecords.end())
      F = CloneRecords.at(F).Src;
    return const_cast<Function *>(F);
  }

  DSGraph *getDSGraph(const Function *F) const {
    return DSA->getDSGraph(*getSrcFunc(F));
  }

  IndexTriple loadTriple(Value *Ptr, Instruction *InsertPt) const;
  void storeTriple(IndexTriple T, Value *Ptr, Instruction *InsertPt) const;

  void applyTransformation(
      Module &M, const TargetMapTy &TransTargets,
      const TargetMapTy &TargetsInGG,
      std::map<unsigned, std::shared_ptr<TransformLowering>> &Transforms);

  bool shouldCloneFunction(Function *F,
                           const RefSetTracker<TargetMapTy> &Targets) const;

  template <typename NODE_SET_TY>
  Type *getNewType(const DSNode *N, const RefSetTracker<NODE_SET_TY> &Refs,
                   Type *CurTy, unsigned Offset);

  template <typename NODE_SET_TY>
  Type *getNewType(Value *V, DSNodeHandle NH,
                   const RefSetTracker<NODE_SET_TY> &Refs);

  template <typename NODE_SET_TY>
  Constant *getNewInitializer(Constant *Init, const DSNode *N, unsigned Offset,
                              const NODE_SET_TY &Targets);

  FunctionType *computeCloneFnTy(Function *,
                                 const RefSetTracker<TargetMapTy> &);
  Function *cloneFunction(Function *Src, FunctionType *FnTy,
                          const TargetMapTy &Targets,
                          std::map<Value *, IndexTriple> &TripleMap,
                          CloneSignature &Sig, bool ReturnsTarget);
  Function *getRealloc(TransformLowering *, Module *);
  std::vector<DSGraph *> SortedSCCs;
  std::map<DSGraph *, unsigned> SCCToOrderMap;
  std::map<DSGraph *, std::vector<CallEdgeTy>> OutgoingEdges;
  void buildMetaGraph(Module &M);

  std::map<std::pair<const Instruction *, const Function *>, DSGraph::NodeMapTy>
      CalleeCallerMappings;
  const DSGraph::NodeMapTy &getCalleeCallerMapping(CallEdgeTy Edge);

  // same as DSGraph::getDSCallSiteForCallSite except
  // it handles insert values
  DSCallSite getDSCallSiteForCallSite(CallSite) const;

  std::map<DSGraph *, DSGraph::NodeMapTy> GToGGMappings;
  const DSGraph::NodeMapTy &getGToGGMapping(DSGraph *G);
  bool isAllocator(const Function *F);
  bool isDeallocator(const Function *F);

  std::vector<Value *> DeadValues;

  template <typename ValTy> void removeValue(ValTy *V) {
    V->removeFromParent();
    DeadValues.push_back(V);
  }

  void cleanupDeadValues() {
    for (auto vi = DeadValues.rbegin(), ve = DeadValues.rend(); vi != ve; ++vi)
      delete *vi;
    DeadValues.clear();
  }

  // given a call to an allocator, insert a new allocation with a different size
  Value *rewriteAlloc(CallSite OldAlloc, Value *SizeInBytes,
                      Instruction *InsertPt, const Twine &Name);

  template <typename NODE_SET_TY>
  StructType *
  computeTypeWithTripleConv(const DSNode *N,
                            const RefSetTracker<NODE_SET_TY> &Refs) const;

  template <typename NODE_SET_TY>
  unsigned
  computeSizeWithTripleConv(const DSNode *N,
                            const RefSetTracker<NODE_SET_TY> &Refs) const;

  //
  // returns if N points a any one of targets *directly*
  //
  template <typename NODE_SET_TY>
  bool refersTargets(const DSNode *N, const NODE_SET_TY &Targets) const {
    for (auto ei = N->edge_begin(), ee = N->edge_end(); ei != ee; ++ei)
      if (Targets.count(ei->second.getNode()))
        return true;
    return false;
  }

  //
  // helper wrapper around TransformLowering::ComputeBytesToAlloc
  // given the original size of an allocation, TotalSize,
  //
  // return the new size (in number of elements) after applying a
  // transformation, Transform,
  //  and the new size (in bytes)
  //
  std::pair<Value *, Value *> computeAllocSize(Value *TotalSize,
                                               const DSNode *Target,
                                               TransformLowering *Transform,
                                               Instruction *InsertPt) const {
    if (TotalSize->getType() != Int64Ty)
      TotalSize = new ZExtInst(TotalSize, Int64Ty, "", InsertPt);

    Value *Size = BinaryOperator::CreateExactUDiv(
              TotalSize, Transform->getOrigSize(), "size", InsertPt),
          *SizeInBytes = Transform->ComputeBytesToAlloc(Size, InsertPt);
    return {Size, SizeInBytes};
  }

public:
  static char ID;
  LayoutTuner() : ModulePass(ID) {
    initializeLayoutTunerPass(*PassRegistry::getPassRegistry());
  };
  virtual void getAnalysisUsage(AnalysisUsage &) const override;
  virtual bool runOnModule(Module &) override;
  StringRef getPassName() const override { return "DataLayoutTuner pass"; }
};

// Determines if the DSGraph 'should' have a node for a given value.
static bool shouldHaveNodeForValue(const Value *V) {
  // Peer through casts
  V = V->stripPointerCasts();

  // Only pointers get nodes
  if (!isa<PointerType>(V->getType()))
    return false;

  // Undef values, even ones of pointer type, don't get nodes.
  if (isa<UndefValue>(V))
    return false;

  if (isa<ConstantPointerNull>(V))
    return false;

  // Use the Aliasee of GlobalAliases
  // FIXME: This check might not be required, it's here because
  // something similar is done in the Local pass.
  if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(V))
    return shouldHaveNodeForValue(GA->getAliasee());

  return true;
}

template <typename SET_TY> void insertInto(SET_TY &Dest, const SET_TY &Src) {
  Dest.insert(Src.begin(), Src.end());
}

void mergeWithNode(FieldsTy &Fields, const DSNode *N, unsigned Offset) {
  for (auto ti = N->type_begin(), te = N->type_end(); ti != te; ++ti) {
    auto *Ty = *ti->second->begin();
    Fields[Offset + ti->first] = Ty;
  }
}

} // end anonymous namespace

char LayoutTuner::ID = 42;

// define 'initializeLayoutTunerPass()'
INITIALIZE_PASS_BEGIN(LayoutTuner, "", "", true, true)
initializeTypeSafetyEquivBU(Registry);
initializeTypeSafetyEQTD(Registry);
INITIALIZE_PASS_DEPENDENCY(EquivBUDataStructures)
INITIALIZE_PASS_DEPENDENCY(AllocIdentify)
INITIALIZE_PASS_END(LayoutTuner, "", "", true, true)

// FIXME this doens't work for fields with union type
// need to get the largest type in this case
TransformLowering::TransformLowering(const DSNode *TheTarget)
    : Target(TheTarget) {
  std::vector<Type *> Fields;
  unsigned i = 0;
  DEBUG(errs() << "TARGET FIELDS:\n");
  for (auto ti = Target->type_begin(), te = Target->type_end(); ti != te;
       ++ti, ++i) {
    Type *FieldTy = *ti->second->begin();
    Fields.push_back(FieldTy);
    Offset2IdxMap[ti->first] = i;
    DEBUG(errs() << "\t(" << ti->first << "): " << *FieldTy << '\n');
  }
  LLVMContext &Ctx = Fields[0]->getContext();
  TargetTy = StructType::get(Ctx, Fields);
  Size = Target->getSize();
  ElemSize = ConstantInt::get(Type::getInt64Ty(Ctx), Target->getSize());
  DEBUG(errs() << "TARGET TYPE: " << *TargetTy
               << " ORIG TARGET SIZE: " << Target->getSize() << '\n');
}

Value *NoopTransform::ComputeAddress(IndexTriple Triple, unsigned Offset,
                                     Instruction *InsertPt, bool DontCare) {
  auto *Int32Ty = Type::getInt32Ty(InsertPt->getContext());
  assert(DontCare ||
         Offset2IdxMap.count(Offset) && "invalid offset used for indexing");
  if (DontCare)
    Offset = Offset2IdxMap.begin()->first;
  auto *FieldIdx = ConstantInt::get(Int32Ty, Offset2IdxMap.at(Offset));
  auto *Base = CastInst::CreatePointerCast(
      Triple.Base, PointerType::getUnqual(TargetTy), "", InsertPt);
  return GetElementPtrInst::Create(
      TargetTy, Base, GEPIdxTy{Triple.Idx, FieldIdx}, "newAddress", InsertPt);
}

LayoutLowering::LayoutLowering(const DSNode *Target,
                               const LayoutDataType *TheLayout,
                               std::string SizeV, std::string IdxV,
                               const DataLayout *TheTD)
    : TransformLowering(Target), TD(TheTD), SizeVar(SizeV), IdxVar(IdxV),
      Layout(TheLayout) {
  // mapping a node in a layout tree -> its direct parent
  std::map<const LayoutDataType *, const LayoutDataType *> Parents;

  // run DFS on the tree
  std::function<void(int, const LayoutDataType *Layout)> Visit =
      [&](int FieldId, const LayoutDataType *Layout) -> void {
    if (auto *Scalar = dyn_cast<LayoutScalar>(Layout)) {
      //
      // reached leaf node in the tree, trace back to construct path for this
      // field
      //
      assert(FieldId >= 0);
      auto &Path = Paths[FieldId];
      assert(Path.empty());
      auto *CurNode = Layout;
      while (CurNode != nullptr) {
        Path.push_back(CurNode);
        CurNode = Parents[CurNode];
      }
      std::reverse(Path.begin(), Path.end());
      return;
    }

    auto *Struct = cast<LayoutStruct>(Layout);
    for (auto &Field : Struct->getFields()) {
      int ChildFieldId = Field.first;
      auto *ChildLayout = Field.second.get();
      Parents[ChildLayout] = Layout;
      Visit(ChildFieldId, ChildLayout);
    }

  };

  Visit(-1, TheLayout);

  assert(Paths.size() == Offset2IdxMap.size());
}

Value *LayoutLowering::ComputeAddress(IndexTriple Triple, unsigned Offset,
                                      Instruction *InsertPt, bool DontCare) {
  assert(DontCare || Offset2IdxMap.count(Offset) && "unknown offset");
  if (DontCare)
    Offset = Offset2IdxMap.begin()->first;
  auto &Path = Paths[Offset2IdxMap.at(Offset)];
  ExprPtr DistFromBase = Const(0);

  const LayoutDataType *CurNode = nullptr;
  unsigned PathIdx = 0, PathLen = Path.size();
  for (; PathIdx < PathLen; PathIdx++) {
    CurNode = Path[PathIdx];

    // we can represent this sublayout using llvm's type
    if (CurNode->getElementType())
      break;

    DistFromBase =
        DistFromBase + CurNode->getSizeForOne(TD) * CurNode->getIdxExpr();

    if (auto *Struct = dyn_cast<LayoutStruct>(CurNode))
      DistFromBase = DistFromBase + Struct->getOffsetOf(Path[PathIdx + 1]);
  }

  assert(CurNode != nullptr);

  auto &Ctx = InsertPt->getContext();
  Type *Int8PtrTy = Type::getInt8PtrTy(Ctx), *Int8Ty = Type::getInt8Ty(Ctx),
       *Int32Ty = Type::getInt32Ty(Ctx);
  auto *Base =
      CastInst::CreatePointerCast(Triple.Base, Int8PtrTy, "", InsertPt);
  std::map<std::string, Value *> Variables{{SizeVar, Triple.Size},
                                           {IdxVar, Triple.Idx}};
  // Damn.. It's hard to name this variable.
  // Anyhow, this is the address from which onward everything is representable
  // by an llvm type.
  // e.g. For `S = { n x a, n x b }; s.b[i]`, we would like to say S.b[i].
  // But can only say `(b*)(S+sizeof(a)*n)[i]`.
  //
  // In this example Addr is `S+sizeof(a)*n`
  auto *Addr = GetElementPtrInst::Create(
      Int8Ty, Base, GEPIdxTy{DistFromBase->emitCode(Variables, InsertPt)}, "",
      InsertPt);

  GEPIdxTy GEPIdxs{CurNode->getIdxExpr()->emitCode(Variables, InsertPt)};
  for (; PathIdx < PathLen; PathIdx++) {
    auto *Node = Path[PathIdx];
    auto Dims = Node->getDims();

    // find idx expression for array indexing
    // Skip this for CurNode for which we've emitted a linearized index
    // expression
    if (Node != CurNode) {
      for (auto DI = Dims.rbegin(), DE = Dims.rend(); DI != DE; ++DI)
        GEPIdxs.push_back(DI->second->emitCode(Variables, InsertPt));
    }

    // find idx for struct offset
    if (auto *Struct = dyn_cast<LayoutStruct>(Node)) {
      int Offset = -1, i = 0;
      auto Fields = Struct->getFields();
      auto *NextNode = Path[PathIdx + 1];
      for (auto &Field : Fields) {
        if (NextNode == Field.second.get()) {
          Offset = i;
          break;
        }
        i++;
      }
      assert(Offset >= 0);
      GEPIdxs.push_back(ConstantInt::get(Int32Ty, Offset));
    }
  }

  auto *ETy = CurNode->getElementType();
  assert(ETy);
  auto *Ptr = CastInst::CreatePointerCast(Addr, PointerType::getUnqual(ETy), "",
                                          InsertPt);
  return GetElementPtrInst::Create(ETy, Ptr, GEPIdxs, "newAddress", InsertPt);
}

Value *LayoutLowering::ComputeBytesToAlloc(Value *Size, Instruction *InsertPt) {
  return Layout->getSizeForAll(TD)->emitCode({{SizeVar, Size}}, InsertPt);
}

//
// Given an allocation, OldAlloc, insert an equivalent allocation with a
// different size, SizeInBytes
//
Value *LayoutTuner::rewriteAlloc(CallSite OldAlloc, Value *SizeInBytes,
                                 Instruction *InsertPt,
                                 const Twine &Name = "") {
  Function *Allocator = OldAlloc.getCalledFunction();

  std::vector<Value *> AllocArgs;
  if (Allocator->getName() == "calloc")
    // fucking bullshit
    AllocArgs = {One64, SizeInBytes};
  else if (Allocator->getName() == "realloc")
    // yet another fucking bullshit
    AllocArgs = {OldAlloc.getArgOperand(0), SizeInBytes};
  else
    AllocArgs = {SizeInBytes};

  Value *NewAlloc;
  assert(Allocator && "can't rewrite indirect memory allocation");
  if (auto *Invoke = dyn_cast<InvokeInst>(OldAlloc.getInstruction()))
    NewAlloc =
        InvokeInst::Create(Allocator, Invoke->getNormalDest(),
                           Invoke->getUnwindDest(), AllocArgs, Name, InsertPt);
  else
    NewAlloc = CallInst::Create(Allocator, AllocArgs, Name, InsertPt);
  return NewAlloc;
}

//
// Figure out type of a node after triple conversion
//
// FIXME: this doesn't work if one of the fields is a union or when dealing with
// packed struct
//
template <typename NODE_SET_TY>
StructType *LayoutTuner::computeTypeWithTripleConv(
    const DSNode *N, const RefSetTracker<NODE_SET_TY> &Refs) const {
  std::vector<Type *> Elems;
  auto *OrigN = Refs.getOrigNode(N);
  for (auto ti = OrigN->type_begin(), te = OrigN->type_end(); ti != te; ++ti) {
    unsigned Offset = ti->first;
    Type *ET = *ti->second->begin();
    if (Refs.hasRefAt(N, Offset))
      Elems.push_back(TripleTy);
    else
      Elems.push_back(ET);
  }
  return StructType::create(Elems);
}

//
// figure out the size of a DSNode that *might* have gone through triple
// conversion
//
//
template <typename NODE_SET_TY>
unsigned LayoutTuner::computeSizeWithTripleConv(
    const DSNode *N, const RefSetTracker<NODE_SET_TY> &Refs) const {

  auto *ConvertedType = computeTypeWithTripleConv(N, Refs);
  return TD->getTypeAllocSize(ConvertedType);
}

Function *LayoutTuner::getRealloc(TransformLowering *Transform, Module *M) {
  const DSNode *N = Transform->getTarget();

  auto ReallocIt = Reallocs.find(N);
  if (ReallocIt != Reallocs.end())
    return ReallocIt->second;

  auto *ReallocTy = FunctionType::get(
      Int8PtrTy, std::vector<Type *>{Int8PtrTy, Int64Ty, Int64Ty},
      false /*varags*/);
  auto *Realloc = Function::Create(ReallocTy, Function::InternalLinkage,
                                   "specialized_realloc", M);
  Realloc->addFnAttr(Attribute::NoInline);

  Constant *Zero = ConstantInt::get(Int64Ty, 0),
           *One = ConstantInt::get(Int64Ty, 1);

  Reallocs[N] = Realloc;
  Value *Old = &*Realloc->arg_begin(),
        *NewSize = &*std::next(Realloc->arg_begin()),
        *OldSize = &*std::next(Realloc->arg_begin(), 2);
  auto &Ctx = Realloc->getContext();

  auto *Entry = BasicBlock::Create(Ctx, "entry", Realloc);
  auto *Ret = BasicBlock::Create(Ctx, "ret", Realloc);
  auto *LoopCond = BasicBlock::Create(Ctx, "loop.cond", Realloc);
  auto *LoopBody = BasicBlock::Create(Ctx, "loop.body", Realloc);

  auto *Noop = new AllocaInst(Int8Ty, "", Entry);
  auto *Bytes = Transform->ComputeBytesToAlloc(NewSize, Noop);
  auto *IntPtrTy = TD->getIntPtrType(Int8PtrTy);
  auto *New =
      CallInst::CreateMalloc(Noop, IntPtrTy, Int8Ty, One, Bytes, nullptr, "");
  auto *Null = ConstantPointerNull::get(Int8PtrTy);
  auto *IsNull = new ICmpInst(*Entry, ICmpInst::ICMP_EQ, Old, Null);
  // if the object being realloc'd is null, we don't need to copy anything
  BranchInst::Create(Ret /*true*/, LoopCond /*false*/, IsNull /*cond*/, Entry);

  // free old obj and return
  auto *RetI = ReturnInst::Create(Ctx, New, Ret);
  CallInst::CreateFree(Old, RetI);

  // loop over Old and copy things to New
  auto *Idx = PHINode::Create(Int64Ty, 2, "i", LoopCond);
  auto *NewIsGreater = new ICmpInst(*LoopCond, ICmpInst::ICMP_UGE, NewSize, OldSize);
  auto *SizeToCopy = SelectInst::Create(NewIsGreater, OldSize, NewSize, "", LoopCond);
  auto *LessThanSize =
      new ICmpInst(*LoopCond, ICmpInst::ICMP_UGE, Idx, SizeToCopy);
  BranchInst::Create(Ret /*true*/, LoopBody /*false*/, LessThanSize /*cond*/,
                     LoopCond);

  auto *IdxInc = BinaryOperator::CreateNSWAdd(Idx, One, "i.inc", LoopBody);
  auto *Continue = BranchInst::Create(LoopCond /*true*/, LoopBody);
  Idx->addIncoming(Zero, Entry);
  Idx->addIncoming(IdxInc, LoopBody);
  // loop over each offset and do copying
  for (auto fi = N->type_begin(), fe = N->type_end(); fi != fe; ++fi) {
    unsigned Offset = fi->first;
    auto *OldAddr = Transform->ComputeAddress({Old, Idx, OldSize}, Offset,
                                              Continue /*insert pt*/);
    auto *OldValue = new LoadInst(OldAddr, "old.val", Continue);
    auto *NewAddr =
        Transform->ComputeAddress({New, Idx, NewSize}, Offset, Continue);
    new StoreInst(OldValue, NewAddr, Continue);
  }

  return Realloc;
}

//
// clone a function and change it to use arguments properly
//
Function *LayoutTuner::cloneFunction(Function *Src, FunctionType *FnTy,
                                     const TargetMapTy &Targets,
                                     std::map<Value *, IndexTriple> &TripleMap,
                                     CloneSignature &Sig, bool ReturnsTarget) {
  ValueToValueMapTy ValueMap;
  auto *Clone = CloneFunction(Src, ValueMap);
  Clone->setVisibility(Function::DefaultVisibility);
  Clone->setLinkage(Function::InternalLinkage);

  auto *NewF = Function::Create(FnTy, Function::InternalLinkage);

  auto &CloneRecord = CloneRecords[NewF];
  CloneCache[Sig] = NewF;

  //
  // map values from clone back to src function
  // note that Src itself could be a clone
  //
  auto It = CloneRecords.find(Src);
  if (It != CloneRecords.end()) {
    CloneRecord.Src = It->second.Src;
    for (auto KV : ValueMap) {
      auto OrigValIt =
          It->second.CloneToSrcValueMap.find(getOrigValue(KV.first));
      if (OrigValIt != It->second.CloneToSrcValueMap.end())
        CloneRecord.CloneToSrcValueMap[KV.second] = OrigValIt->second;
    }
  } else {
    for (auto KV : ValueMap)
      CloneRecord.CloneToSrcValueMap[KV.second] = getOrigValue(KV.first);
    CloneRecord.Src = Src;
  }

  // TODO: clone attributes from the clone
  Clone->getParent()->getFunctionList().insert(Clone->getIterator(), NewF);
  NewF->setName("CLONED." + Clone->getName());
  NewF->getBasicBlockList().splice(NewF->begin(), Clone->getBasicBlockList());

  if (Src->hasPersonalityFn())
    NewF->setPersonalityFn(Src->getPersonalityFn());

  // transfer arguments from Clone to NewF
  auto I2 = NewF->arg_begin();
  if (ReturnsTarget)
    I2 = std::next(I2, 3);
  for (auto I = Clone->arg_begin(), E = Clone->arg_end(); I != E; ++I, ++I2) {
    auto *N = getNodeForValue(&*I, NewF).getNode();
    if (Targets.count(N)) {
      // pointer to target has been expanded into triple
      auto *Base = &*I2++;
      auto *Idx = &*I2++;
      auto *Size = &*I2;
      DEBUG(errs() << "Setting triple for " << &*I << "\n");
      Base->setName("base");
      Idx->setName("idx");
      Size->setName("size");
      TripleMap[&*I] = {Base, Idx, Size};
    } else {
      CloneRecord.CloneToSrcValueMap[&*I2] =
          CloneRecord.CloneToSrcValueMap[&*I];
      auto *OldTy = I->getType();
      I->mutateType(I2->getType());
      I->replaceAllUsesWith(&*I2);
      I->mutateType(OldTy);
      I2->takeName(&*I);
      updateValue(&*I, &*I2);
    }
  }

  removeValue(Clone);
  return NewF;
}

// TODO: support c++ allocators
bool LayoutTuner::isAllocator(const Function *F) {
  StringRef Name = F->getName();
  return Name == "malloc" || Name == "calloc" || Name == "_Znwm" ||
         Name == "_Znam" || Name == "_Znwj" || Name == "_Znaj" ||
         Name == "realloc";
}

// TODO: support c++ deallocators
bool LayoutTuner::isDeallocator(const Function *F) {
  StringRef Name = F->getName();
  return Name == "free" || Name == "_ZdlPv" || Name == "_ZdaPv";
}

bool LayoutTuner::runOnModule(Module &M) {
  AllocIdentifier = &getAnalysis<AllocIdentify>();
  DSA = &getAnalysis<EquivBUDataStructures>();
  TD = &DSA->getDataLayout();

  LLVMContext &Ctx = M.getContext();

  VoidTy = Type::getVoidTy(Ctx);
  Int8Ty = Type::getInt8Ty(Ctx);
  Int8PtrTy = Type::getInt8PtrTy(Ctx);
  Int32Ty = Type::getInt32Ty(Ctx);
  Int64Ty = Type::getInt64Ty(Ctx);
  std::vector<Type *> TripleElems = {Int8PtrTy, Int64Ty, Int64Ty};
  TripleTy = StructType::get(Ctx, TripleElems);
  TriplePtrTy = PointerType::getUnqual(TripleTy);

  Zero = ConstantInt::get(Int32Ty, 0);
  One = ConstantInt::get(Int32Ty, 1);
  Two = ConstantInt::get(Int32Ty, 2);
  Zero64 = ConstantInt::get(Int64Ty, 0);
  One64 = ConstantInt::get(Int64Ty, 1);

  buildMetaGraph(M);

  auto Targets = findLegalTargets(M);
  errs() << "Number of unique nodes safe to transform: " << Targets.size()
         << '\n';

  // mappping trasform group -> the group's layout
  std::map<unsigned, std::shared_ptr<TransformLowering>> Transforms;

  TargetMapTy TargetsInGG;

  TransformPool TP;
  TP.addTransform(make_unique<FactorTransform>(4), 0.05);      
  TP.addTransform(make_unique<StructTransform>(), 0.05);        // 0.1
  TP.addTransform(make_unique<StructFlattenTransform>(), 0.05); // 0.15
  TP.addTransform(make_unique<SOATransform>(), 0.3);            // 0.45
  TP.addTransform(make_unique<AOSTransform>(), 0.3);            // 0.75
  TP.addTransform(make_unique<SwapTransform>(), 0.2);           // 0.95
  TP.addTransform(make_unique<InterchangeTransform>(), 0.05);   // 1.00

  unsigned i = 0;
  auto *GG = DSA->getGlobalsGraph();
  // mapping group -> variables for <size, idx>
  std::map<unsigned, std::pair<std::string, std::string>> Variables;
  LayoutSet OrigLayouts;
  std::map<unsigned, const DSNode *> Group2Target;
  for (auto GroupAndTarget : Targets) {
    const DSNode *Target = GroupAndTarget.first;
    unsigned Group = GroupAndTarget.second;
    if (Target->getParentGraph() == GG)
      TargetsInGG[Target] = Group;
    auto SizeVar = getNewName(), IdxVar = getNewName();
    OrigLayouts.emplace_back(Group,
                             LayoutStruct::create(Target, TD, SizeVar, IdxVar));
    Variables[Group] = {SizeVar, IdxVar};
    Group2Target[Group] = Target;
  }

  // set transforms based on layout
  auto setTransforms = [&](LayoutSet Layouts) {
    for (auto &GroupAndLayout : Layouts) {
      unsigned Group;
      const LayoutDataType *Layout;
      std::string SizeVar, IdxVar;
      std::tie(Group, Layout) = GroupAndLayout;
      std::tie(SizeVar, IdxVar) = Variables.at(Group);
      Transforms[Group].reset(new LayoutLowering(Group2Target.at(Group), Layout,
                                                 SizeVar, IdxVar, TD));
    }
  };

  TransformState State(OrigLayouts, &TP, MutateProb, ResetProb);
  double CurCost = std::numeric_limits<double>::max();
  double T = TMax, Alpha = std::pow(TMin / TMax, 1.0 / double(Iterations));
  errs() << "T = " << T << ", ALPHA = " << Alpha << '\n';
  for (int i = 0; i < Iterations; i++) {
    setTransforms(State.getLayouts());

    // set up a pipe for the child process to report back
    // performance of the transformed code
    int Pipe[2], &PipeIn = Pipe[0], &PipeOut = Pipe[1];
    pipe(Pipe);
    int Child = fork();
    switch (Child) {
    case -1:
      errs() << "Failed to fork\n";
      exit(1);

    case 0: {
      // child process
      // apply transformation, evaluate the layout and write cost to pipe
      close(PipeIn);
      applyTransformation(M, Targets, TargetsInGG, Transforms);
      cleanupDeadValues();
      double Cost = evaluate(M, TestArgv);
      ssize_t BytesWritten = write(PipeOut, &Cost, sizeof Cost);
      close(PipeOut);
      if (BytesWritten < 0) {
        errs() << "Failed to write to pipe\n";
        exit(1);
      }
      exit(0);
    }

    default: {
      // parent process
      double Cost;
      close(PipeOut);
      ssize_t BytesRead = read(PipeIn, &Cost, sizeof Cost);
      close(PipeIn);
      int Stat;
      waitpid(Child, &Stat, 0);
      if (BytesRead < 0) {
        errs() << "Failed to read from pipe\n";
        exit(1);
      } else if (BytesRead == sizeof Cost && Cost > 0) {
        State.setCost(Cost);
      } else {
        errs() << "Failed to evaluate layout\n";
        for (auto &GroupAndLayout : State.getLayouts()) {
          errs() << "LAYOUT DUMP, Group = " << GroupAndLayout.first << ": ";
          GroupAndLayout.second->dump();
          errs() << '\n';
        }
      }

      double AcceptProb =
                 (BytesRead == sizeof Cost && Cost > 0
                      ? std::min<double>(std::exp((CurCost - Cost) / T), 1)
                      : 0),
             P = randProb();
      errs() << "new = " << Cost << ", cur = " << CurCost
             << ", accept prob = " << AcceptProb << ", p = " << P << '\n';
      bool Accept = P < AcceptProb;
      if (!Accept)
        State.revert();
      else
        CurCost = Cost;
    }
    }

    State.mutate();

    T *= Alpha;
  }

  setTransforms(State.getBest());
  for (auto &GroupAndLayout : State.getBest()) {
    errs() << "Layout (" << GroupAndLayout.first << "): ";
    GroupAndLayout.second->dump();
    errs() << '\n';
  }
  applyTransformation(M, Targets, TargetsInGG, Transforms);
  if (Dump)
    M.dump();
  cleanupDeadValues();
  CalleeCallerMappings.clear();
  GToGGMappings.clear();

  return true;
}

// side-effect:
// 1) populate SortedSCCs
// 2) populate OutgoingEdges
// 3) populate SCCToOrderMap
void LayoutTuner::buildMetaGraph(Module &M) {
  SortedSCCs.clear();
  OutgoingEdges.clear();

  const DSCallGraph *CG = &DSA->getCallGraph();

  for (auto ki = CG->key_begin(), ke = CG->key_end(); ki != ke; ++ki) {
    CallSite CS = *ki;
    Function *Caller = CS.getParent()->getParent();
    auto *CallerG = DSA->getDSGraph(*Caller);

    //
    // direct call
    //
    if (auto *Callee = CS.getCalledFunction()) {
      if (!Callee->empty() && DSA->getDSGraph(*Callee) != CallerG)
        OutgoingEdges[DSA->getDSGraph(*Caller)].push_back({CS, Callee});
      continue;
    }

    //
    // indirect call
    //
    for (auto ci = CG->callee_begin(CS), ce = CG->callee_end(CS); ci != ce;
         ++ci) {
      const Function *Callee = *ci;
      if (!Callee->empty() && DSA->getDSGraph(*Callee) != CallerG)
        OutgoingEdges[DSA->getDSGraph(*Caller)].push_back({CS, Callee});
    }
  }

  // DFS
  std::set<DSGraph *> Visited;
  std::function<void(DSGraph *)> Visit = [&](DSGraph *G) {
    if (!Visited.insert(G).second)
      return;

    auto &Edges = OutgoingEdges[G];

    // Visit adjacent SCCs
    for (auto &Edge : Edges) {
      DSGraph *Neighbor = DSA->getDSGraph(*Edge.second);
      Visit(Neighbor);
    }

    SortedSCCs.push_back(G);
  };

  for (auto &F : M)
    if (!F.empty())
      Visit(DSA->getDSGraph(F));

  std::reverse(SortedSCCs.begin(), SortedSCCs.end());
  unsigned i = 0;
  for (auto *G : SortedSCCs)
    SCCToOrderMap[G] = i++;
}

template <typename NODE_SET_TY>
Type *LayoutTuner::getNewType(const DSNode *N,
                              const RefSetTracker<NODE_SET_TY> &Refs,
                              Type *CurTy, unsigned Offset) {
  // give up if we can't analyze `N`
  if (!N || N->isCollapsedNode() || Offset >= N->getSize()) {
    return CurTy;
  }

  // end of recursion -- if a pointer points to the allocation we want to
  // transform
  // we simply expand it into a triple
  if (Refs.getTargets().count(N))
    return TripleTy;

  if (!Refs.refersTargets(N))
    return CurTy;

  if (auto *ST = dyn_cast<StructType>(CurTy)) {
    auto *SL = TD->getStructLayout(ST);
    std::vector<Type *> Elements;

    for (unsigned i = 0; i < ST->getNumElements(); i++) {
      auto *ET = ST->getElementType(i);
      unsigned ElementOffset = Offset + SL->getElementOffset(i);

      if (ElementOffset >= N->getSize()) {
        Elements.push_back(ET);
        continue;
      }

      Type *NewET;
      if (ET->isPointerTy()) {
        if (Refs.hasRefAt(N, ElementOffset))
          NewET = TripleTy;
        else
          NewET = ET;
      } else
        NewET = getNewType(N, Refs, ET, ElementOffset);

      Elements.push_back(NewET);
    }

    return StructType::get(CurTy->getContext(), Elements);
  }

  if (auto *PT = dyn_cast<PointerType>(CurTy)) {
    auto *ET = PT->getElementType();
    Type *NewPointeeTy;
    if (isa<StructType>(ET))
      NewPointeeTy = getNewType(N, Refs, ET, Offset);
    else if (Refs.hasRefAt(N, Offset))
      NewPointeeTy = TripleTy;
    else if (N->hasLink(Offset)) {
      DSNodeHandle PointeeNH = N->getLink(Offset);
      NewPointeeTy =
          getNewType(PointeeNH.getNode(), Refs, ET, PointeeNH.getOffset());
    } else
      NewPointeeTy = ET;

    return PointerType::getUnqual(NewPointeeTy);
  }

  if (auto *AT = dyn_cast<ArrayType>(CurTy)) {
    auto *NewET = getNewType(N, Refs, AT->getElementType(), Offset);
    return ArrayType::get(NewET, AT->getNumElements());
  }

  return CurTy;
}

// get new type of a value after pointer conversion
template <typename NODE_SET_TY>
Type *LayoutTuner::getNewType(Value *V, DSNodeHandle NH,
                              const RefSetTracker<NODE_SET_TY> &Refs) {
  if (NH.isNull())
    return V->getType();

  unsigned Offset = NH.getOffset();
  auto *N = NH.getNode();

  return getNewType(N, Refs, getOrigType(V), Offset);
}

//
// rewrite a global initialier that points to target
//
template <typename NODE_SET_TY>
Constant *LayoutTuner::getNewInitializer(Constant *Init, const DSNode *N,
                                         unsigned Offset,
                                         const NODE_SET_TY &Targets) {
  if (!Init)
    return nullptr;

  if (refersTargets(N, Targets)) {
    assert(isa<ConstantPointerNull>(Init) &&
           "Unsupported use of global reference to target");
    std::vector<Constant *> Triple = {ConstantPointerNull::get(Int8PtrTy),
                                      Zero64, Zero64};
    return ConstantStruct::get(TripleTy, Triple);
  }

  assert(isa<ConstantStruct>(Init) ||
         isa<ConstantArray>(Init) &&
             "Unsupported use of global reference to target");

  if (isa<ConstantStruct>(Init)) {
    llvm_unreachable("not implemented");
  } else { // constant array
    llvm_unreachable("not implemented");
  }
}

void LayoutTuner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TypeSafety<EquivBUDataStructures>>();
  AU.addRequired<EquivBUDataStructures>();
  AU.addRequired<AllocIdentify>();
}

IndexTriple LayoutTuner::loadTriple(Value *Ptr, Instruction *InsertPt) const {
  // load base ptr
  auto *BaseAddr = GetElementPtrInst::Create(
      TripleTy, Ptr, GEPIdxTy{Zero, Zero}, "base_addr", InsertPt);
  auto *Base = new LoadInst(BaseAddr, "base", InsertPt);

  // load idx
  auto *IdxAddr = GetElementPtrInst::Create(TripleTy, Ptr, GEPIdxTy{Zero, One},
                                            "idx_addr", InsertPt);
  auto *Idx = new LoadInst(IdxAddr, "Idx", InsertPt);

  // load offset
  auto *SizeAddr = GetElementPtrInst::Create(TripleTy, Ptr, GEPIdxTy{Zero, Two},
                                             "size_addr", InsertPt);
  auto *Size = new LoadInst(SizeAddr, "size", InsertPt);

  return {Base, Idx, Size};
}

void LayoutTuner::storeTriple(IndexTriple T, Value *Ptr,
                              Instruction *InsertPt) const {
  auto *BaseAddr = GetElementPtrInst::Create(
      TripleTy, Ptr, GEPIdxTy{Zero, Zero}, "base_addr", InsertPt);
  new StoreInst(T.Base, BaseAddr, InsertPt);

  auto *IdxAddr = GetElementPtrInst::Create(TripleTy, Ptr, GEPIdxTy{Zero, One},
                                            "idx_addr", InsertPt);
  new StoreInst(T.Idx, IdxAddr, InsertPt);

  auto *SizeAddr = GetElementPtrInst::Create(TripleTy, Ptr, GEPIdxTy{Zero, Two},
                                             "size_addr", InsertPt);
  new StoreInst(T.Size, SizeAddr, InsertPt);
}

const DSGraph::NodeMapTy &LayoutTuner::getGToGGMapping(DSGraph *G) {
  auto It = GToGGMappings.find(G);
  if (It != GToGGMappings.end())
    return It->second;

  auto &Mapping = GToGGMappings[G];
  G->computeGToGGMapping(Mapping);
  return Mapping;
}

DSCallSite LayoutTuner::getDSCallSiteForCallSite(CallSite CS) const {
  DSNodeHandle RetVal, VarArg;
  Instruction *I = CS.getInstruction();
  Function *F = I->getParent()->getParent();
  if (shouldHaveNodeForValue(I))
    RetVal = getNodeForValue(I, F);

  // FIXME: Here we trust the signature of the callsite to determine which
  // arguments
  // are var-arg and which are fixed.  Apparently we can't assume this, but I'm
  // not sure
  // of a better way.  For now, this assumption is known limitation.
  const FunctionType *CalleeFuncType = DSCallSite::FunctionTypeOfCallSite(CS);
  int NumFixedArgs = CalleeFuncType->getNumParams();

  // Sanity check--this really, really shouldn't happen
  if (!CalleeFuncType->isVarArg())
    assert(CS.arg_size() == static_cast<unsigned>(NumFixedArgs) &&
           "Too many arguments/incorrect function signature!");

  std::vector<DSNodeHandle> Args;
  Args.reserve(CS.arg_end() - CS.arg_begin());

  // Calculate the arguments vector...
  for (CallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end(); I != E; ++I)
    if (isa<PointerType>((*I)->getType())) {
      DSNodeHandle ArgNode; // Initially empty
      if (shouldHaveNodeForValue(*I))
        ArgNode = getNodeForValue(I->get(), F);
      if (I - CS.arg_begin() < NumFixedArgs) {
        Args.push_back(ArgNode);
      } else {
        VarArg.mergeWith(ArgNode);
      }
    }

  //
  // Add a new function call entry.  We get the called value from the call site
  // and strip pointer casts instead of asking the CallSite class to do that
  // since CallSite::getCalledFunction() returns 0 if the called value is
  // a bit-casted function constant.
  //
  if (Function *Callee =
          dyn_cast<Function>(CS.getCalledValue()->stripPointerCasts()))
    return DSCallSite(CS, RetVal, VarArg, Callee, Args);
  else
    return DSCallSite(CS, RetVal, VarArg,
                      getNodeForValue(CS.getCalledValue(), F).getNode(), Args);
}

//
// Wrapper around DSGraph::computeCalleeCallerMapping to cache reuslt
//
const DSGraph::NodeMapTy &
LayoutTuner::getCalleeCallerMapping(LayoutTuner::CallEdgeTy Edge) {
  CallSite CS;
  const Function *Callee;

  std::tie(CS, Callee) = Edge;
  Function *Caller = CS.getParent()->getParent();
  const Instruction *I = cast<Instruction>(getOrigValue(CS.getInstruction()));
  auto CloneIt = CloneRecords.find(Caller);
  if (CloneIt != CloneRecords.end()) {
    assert(CloneIt->second.CloneToSrcValueMap.count(I));
    I = cast<Instruction>(CloneIt->second.CloneToSrcValueMap.at(I));
    Caller = CloneIt->second.Src;
  }

  Callee = getSrcFunc(Callee);

  auto CacheKey = std::make_pair(I, Callee);
  auto MappingIt = CalleeCallerMappings.find(CacheKey);
  if (MappingIt != CalleeCallerMappings.end())
    return MappingIt->second;

  auto *G = DSA->getDSGraph(*Caller);

  auto &CalleeCallerMapping = CalleeCallerMappings[CacheKey];
  // TODO: All the work before to get the original value of I is redundant
  DSCallSite DSCS =
      getDSCallSiteForCallSite(CallSite(const_cast<Instruction *>(I)));
  DSGraph *CalleeG = DSA->getDSGraph(*Callee);

  bool Strict = CS.getCalledFunction() != nullptr;
  G->computeCalleeCallerMapping(DSCS, *Callee, *CalleeG, CalleeCallerMapping,
                                false);

  return CalleeCallerMapping;
}

//
// i am too lazy to support these now
//
// TODO: handle memmove, and memset
// TODO: try *really hard* to think about why the way we processing function
// works (I think?)
// TODO: emit alias.scope metadata for transformed loads/stores.
//       Two memory accesses don't alias if
//          1) they are associated with two targets
//       or 2) they have different offsets
// -- push orig functions on to worklist first, and don't process functions that
// should be clone
//
// FIXME: Should record size of target/target-ref on top level, since once once
// one descend down the call graph, DSNode::getSize is not accurate. E.g
// size of a node is likely to be zero in a malloc wrapper
//
void LayoutTuner::applyTransformation(
    Module &M, const TargetMapTy &TransTargets, const TargetMapTy &TargetsInGG,
    std::map<unsigned, std::shared_ptr<TransformLowering>> &Transforms) {

  // mapping global that points to targets -> same globals re-declared with
  // proper types
  std::map<Value *, Value *> RedeclaredGlobals;

  // re-declare globals that points to targets with appropriate type
  auto *GG = DSA->getGlobalsGraph();
  auto gi = M.global_begin(), ge = M.global_end(), next_gi = std::next(gi);
  for (; gi != ge; gi = next_gi, ++next_gi) {
    GlobalVariable &G = *gi;
    auto NH = GG->getNodeForValue(&G);
    auto *N = NH.getNode();
    if (N && refersTargets(N, TargetsInGG)) {
      auto *NewType =
               getNewType(&G, NH, RefSetTracker<TargetMapTy>(TargetsInGG)),
           *NewValType = cast<PointerType>(NewType)->getElementType();
      auto *NewInitializer =
          getNewInitializer(G.getInitializer(), N, NH.getOffset(), TargetsInGG);
      auto *NewG = new GlobalVariable(M, NewValType, G.isConstant(),
                                      G.getLinkage(), NewInitializer, "", &G);
      RedeclaredGlobals[&G] = NewG;
      updateValue(&G, NewG);
    }
  }

  struct RewriterContext {
    Function *F;
    RefSetTracker<TargetMapTy> Refs;
    OffsetMapTy Offsets;
  };
  std::vector<RewriterContext> Worklist;

  OffsetMapTy DefaultOffsets;
  for (auto Pair : TransTargets)
    DefaultOffsets[Pair.first] = 0;
  RewriterContext DefaultContext;
  DefaultContext.Offsets = DefaultOffsets;
  DefaultContext.Refs = TransTargets;

  // scan the call graph top-down and rewrite functions along the way
  for (auto gi = SortedSCCs.rbegin(), ge = SortedSCCs.rend(); gi != ge; ++gi) {
    DSGraph *G = *gi;
    for (auto ri = G->retnodes_begin(), re = G->retnodes_end(); ri != re;
         ++ri) {
      auto InitContext = DefaultContext;
      InitContext.F = const_cast<Function *>(ri->first);
      Worklist.push_back(InitContext);
    }
  }

  // Mapping <pointer to TransTarget> -> <triple converted pointer>
  std::map<Value *, IndexTriple> TripleMap;

  struct IncomingTripleTy {
    PHINode *IncomingBase;
    PHINode *IncomingIdx;
    PHINode *IncomingSize;
    BasicBlock *IncomingBB;
  };

  std::map<Function *, bool> Processed;

  IndexTriple NullTriple{ConstantPointerNull::get(Int8PtrTy),
                         UndefValue::get(Int64Ty), UndefValue::get(Int64Ty)};

  while (!Worklist.empty()) {
    auto &RewriteCtx = Worklist.back();
    Function *F = RewriteCtx.F;
    auto Refs = std::move(RewriteCtx.Refs);
    auto &Targets = Refs.getTargets();
    auto Offsets = std::move(RewriteCtx.Offsets);
    Worklist.pop_back();

    DEBUG(errs() << "!! processing " << F->getName() << '\n');
    DEBUG(errs() << "!!! there are " << Targets.size() << " targets\n");

    if (F->empty())
      continue;

    if (Processed[F])
      continue;
    Processed[F] = true;

    // propagate targets from global graph
    auto &GToGGMapping = getGToGGMapping(getDSGraph(F));
    for (auto &Mapping : GToGGMapping) {
      auto It = TargetsInGG.find(Mapping.second.getNode());
      if (It != TargetsInGG.end() && !Targets.count(Mapping.first)) {
        Offsets[Mapping.first] = 0;
        Targets[Mapping.first] = It->second;
      }
    }
    if (CloneRecords.count(F)) {
      for (auto AI = F->arg_begin(), AE = F->arg_end(); AI != AE; ++AI) {
        auto *Arg = &*AI;
        auto *ArgN = getNodeForValue(Arg, F).getNode();
        if (Targets.count(ArgN)) {
          DEBUG(errs() << "SETTING TRIPLE FOR " << Arg << '\n');
          TripleMap[Arg] = NullTriple;
        }
      }
    }

    // F will be dead after cloning anyways
    if (!CloneRecords.count(F) && shouldCloneFunction(F, Refs))
      continue;

    //
    // Mapping a value to a set of <phi, basic block>, where it's used
    // we remember values whose triples we need when we emit phis but are
    // not computed at the time. We go back to fix them up later.
    //
    std::map<Value *, std::vector<IncomingTripleTy>> IncomingValues;

    // make sure we visit the definition of a value before its use
    // with exception to PHIs
    for (BasicBlock *BB : ReversePostOrderTraversal<Function *>(F)) {
      auto I = BB->begin(), E = BB->end(), NextIt = std::next(I);
      for (; I != E; I = NextIt, ++NextIt) {
        auto NH = getNodeForValue(&*I, F);
        auto *N = NH.getNode();

        for (unsigned i = 0, e = I->getNumOperands(); i != e; i++) {
          auto &Op = I->getOperandUse(i);
          auto Redeclared = RedeclaredGlobals.find(Op.get());
          if (Redeclared != RedeclaredGlobals.end())
            Op.set(Redeclared->second);
        }

        auto *GEP = dyn_cast<GetElementPtrInst>(&*I);
        if (GEP && N) {
          DEBUG(errs() << "--- handling gep\n");
          Value *Src = GEP->getPointerOperand();
          Type *OrigSrcTy = GEP->getSourceElementType();
          Type *SrcTy = getNewType(Src, getNodeForValue(Src, F), Refs);
          DEBUG(errs() << "--- Deduced type for gep\n");

          if (Targets.count(N)) {
            DEBUG(errs() << "--- updating idx\n");
            // Src is Target
            // we postpone the actual address calculation by updating the triple
            assert(isa<ConstantPointerNull>(Src) || TripleMap.count(Src));
            IndexTriple OldTriple;
            if (!isa<ConstantPointerNull>(Src))
              OldTriple = TripleMap.at(Src);
            else
              OldTriple = NullTriple;
            IndexTriple &NewTriple = TripleMap[GEP] = OldTriple;
            Instruction *InsertPt = &*std::next(I);

            unsigned GroupId = Targets.lookup(N);

            // Update Idx
            if (TD->getTypeAllocSize(OrigSrcTy) ==
                Transforms.at(GroupId)->getSize()) {
              // FIXME this is broken
              // -- consider this example `gep {[1 x i32]}`
              // This should be fixed fairly easily by implementing something
              // similar
              // to accumulateConstantOffset to linearize the address
              // computation
              // and divide it by size of the target
              auto *Diff = GEP->idx_begin()->get();
              if (Diff->getType() != Int64Ty)
                Diff = new SExtInst(Diff, Int64Ty, "", InsertPt);
              NewTriple.Idx = BinaryOperator::CreateNSWAdd(
                  OldTriple.Idx, Diff, "updated_idx", InsertPt);
            } else {
              auto SrcNH = getNodeForValue(Src, F);
              APInt RelOffset(TD->getPointerSizeInBits(GEP->getAddressSpace()),
                              0);
              bool HasConstOffset =
                  GEP->accumulateConstantOffset(*TD, RelOffset);
              assert(HasConstOffset);
              unsigned GroupId = Targets.lookup(N);
              Constant *TargetSize = Transforms.at(GroupId)->getOrigSize(),
                       *AbsOffset = ConstantInt::get(
                           TargetSize->getType(),
                           RelOffset + Offsets.lookup(SrcNH.getNode()) +
                               SrcNH.getOffset());
              auto *Diff = BinaryOperator::CreateUDiv(AbsOffset, TargetSize, "",
                                                     InsertPt);
              NewTriple.Idx = BinaryOperator::CreateNSWAdd(
                  OldTriple.Idx, Diff, "updated_idx", InsertPt);
            }

            removeValue(GEP);
          } else if (Src->getType() != SrcTy || Refs.refersTargets(N)) {
            auto *SrcFixed =
                CastInst::CreatePointerCast(Src, SrcTy, "cast4gep", GEP);
            std::vector<Value *> Idxs(GEP->idx_begin(), GEP->idx_end());
            auto *ResultTy =
                GetElementPtrInst::getGEPReturnType(SrcFixed, Idxs);
            GEP->setSourceElementType(
                cast<PointerType>(SrcTy)->getElementType());
            GEP->setResultElementType(
                cast<PointerType>(ResultTy)->getElementType());
            updateValue(Src, SrcFixed);
            GEP->getOperandUse(0).set(SrcFixed);
            updateType(GEP, ResultTy);
          }

        } // end of handling GEP

        auto *SL = dyn_cast<SelectInst>(&*I);
        if (SL && Targets.count(N)) {
          DEBUG(errs() << "--- handling select\n");
          IndexTriple TrueTriple, FalseTriple;
          if (!isa<ConstantPointerNull>(SL->getTrueValue()))
            TrueTriple = TripleMap.at(SL->getTrueValue());
          else
            TrueTriple = NullTriple;
          if (!isa<ConstantPointerNull>(SL->getFalseValue()))
            FalseTriple = TripleMap.at(SL->getFalseValue());
          else
            FalseTriple = NullTriple;
          auto &Triple = TripleMap[SL];
          auto *Cond = SL->getCondition();
          Triple.Base = SelectInst::Create(Cond, TrueTriple.Base,
                                           FalseTriple.Base, "base", &*I);
          Triple.Idx = SelectInst::Create(Cond, TrueTriple.Idx, FalseTriple.Idx,
                                          "idx", &*I);
          Triple.Size = SelectInst::Create(Cond, TrueTriple.Size,
                                           FalseTriple.Size, "size", &*I);
          removeValue(SL);
        } // end of handling select

        if (auto *LI = dyn_cast<LoadInst>(&*I)) {
          DEBUG(errs() << "--- handling load\n");
          if (Targets.count(N)) {
            auto SrcNH = getNodeForValue(LI->getPointerOperand(), F);
            assert(Refs.hasRefAt(SrcNH.getNode(), SrcNH.getOffset()));
            DEBUG(errs() << "--- loading triple-converted pointer\n");
            // case 1: loading pointer to Target
            // instead of loading the pointer, we load the triple
            Instruction *InsertPt = &*std::next(I);
            DEBUG(errs() << "--- Setting triple for " << LI << '\n');
            auto *TriplePtr = CastInst::CreatePointerCast(
                LI->getPointerOperand(), TriplePtrTy, "", InsertPt);
            TripleMap[LI] = loadTriple(TriplePtr, InsertPt);
            removeValue(LI);
          }

          auto PointerNH = getNodeForValue(LI->getPointerOperand(), F);
          if (Targets.count(PointerNH.getNode())) {
            DEBUG(errs() << "--- loading from triple-converted pointer\n");
            // case 2: loading from Target itself
            // we need to replace the load, since layout of the allocation has
            // changed
            //
            // assuming the data structure is not recursive,
            // we forward address calculation of to whatever transformation we
            // are trying to apply,
            // which knows the precise layout
            Instruction *InsertPt = &*std::next(I);

            unsigned GroupId = Targets.lookup(PointerNH.getNode());
            unsigned Offset =
                Offsets.lookup(PointerNH.getNode()) + PointerNH.getOffset();
            DEBUG(errs() << "-- replacing load\n");
            assert(Transforms.count(GroupId) && "transform not found");
            assert(TripleMap.count(LI->getPointerOperand()) &&
                   "triple not found\n");
            auto *Addr = Transforms.at(GroupId)->ComputeAddress(
                TripleMap.at(LI->getPointerOperand()), Offset, InsertPt);
            auto *Ptr = CastInst::CreatePointerCast(
                Addr, LI->getPointerOperand()->getType(), I->getName(),
                InsertPt);
            updateValue(LI->getPointerOperand(), Addr);
            updateValue(LI->getPointerOperand(), Ptr);
            auto *NewLoad = new LoadInst(Ptr, "transformedLoad", InsertPt);
            assert(NewLoad->getType() == LI->getType());
            LI->replaceAllUsesWith(NewLoad);
            updateValue(LI, NewLoad);
            removeValue(LI);
          }
        } // end of handling load

        if (auto *SI = dyn_cast<StoreInst>(&*I)) {
          DEBUG(errs() << "--- handling store\n");
          Value *Dest = SI->getPointerOperand();
          auto DestNH = getNodeForValue(Dest, F);
          auto *DestN = DestNH.getNode();

          Value *Src = SI->getValueOperand();
          auto SrcNH = getNodeForValue(Src, F);
          auto *SrcN = SrcNH.getNode();

          if (Targets.count(SrcN)) {
            // case 1: storing pointer to Target
            //
            // we replace original store of pointer to store of the triple
            Instruction *InsertPt = &*std::next(I);
            auto *TriplePtr =
                CastInst::CreatePointerCast(Dest, TriplePtrTy, "", InsertPt);
            assert(TripleMap.count(Src));
            storeTriple(TripleMap.at(Src), TriplePtr, InsertPt);
            removeValue(SI);
          } else if (Targets.count(DestN)) {
            // case 2: storing to target
            // we need to replace the store, since layout of the allocation has
            // changed
            //
            // assuming the data structure is not recursive,
            // we forward address calculation of to whatever transformation we
            // are trying to apply,
            // which knows the precise layout
            Instruction *InsertPt = &*std::next(I);
            unsigned GroupId = Targets.lookup(DestN);
            assert(Offsets.count(DestN) && "offset not found for dest node");
            unsigned Offset = Offsets.lookup(DestN) + DestNH.getOffset();
            auto *Addr = Transforms.at(GroupId)->ComputeAddress(
                TripleMap.at(Dest), Offset, InsertPt);
            auto *Ptr = CastInst::CreatePointerCast(
                Addr, PointerType::getUnqual(Src->getType()), I->getName(),
                InsertPt);
            updateValue(Dest, Addr);
            updateValue(Dest, Ptr);
            auto *NewStore = new StoreInst(Src, Ptr, InsertPt);
            removeValue(SI);
          } else if (DestN && Refs.hasRefAt(DestN, DestNH.getOffset())) {
            // 3) storing to a pointer of target but the the value
            // being stored is not target itself. In this case we have to assume
            // it's a null being stored
            assert(isa<ConstantPointerNull>(Src) && "WTF");
            auto *TriplePtr =
                CastInst::CreatePointerCast(Dest, TriplePtrTy, "", &*I);
            IndexTriple NullTriple{ConstantPointerNull::get(Int8PtrTy),
                                   UndefValue::get(Int64Ty),
                                   UndefValue::get(Int64Ty)};
            storeTriple(NullTriple, TriplePtr, &*I);
            removeValue(&*I);
          } else {
            auto *DestTy = cast<PointerType>(Dest->getType())->getElementType();
            //if (!DestN)  {
            //  const DSNode *X  = DSA->getGlobalsGraph()->getNodeForValue(getOrigValue(Dest)).getNode(),
            //        *Y = nullptr;
            //  auto &GToGGMapping = getGToGGMapping(getDSGraph(F));
            //  for (auto &Mapping : GToGGMapping)
            //    if (Mapping.second.getNode() == X) {
            //      Y = Mapping.first;
            //      break;
            //    }
            //  errs() << "[BS2] casting " << *Src << " -> " << *DestTy
            //    << "\n\t DestN: " << DestN
            //    << "\n\t storing to ref?: " << (DestN && Refs.hasRefAt(DestN, DestNH.getOffset()))
            //    << "\n\t DEST = " << *Dest
            //    << "\n\t OrigDest: " << *getOrigValue(Dest)
            //    << "\n\t F cloned?: " << CloneRecords.count(F)
            //    << "\n\t F = "  << F->getName()
            //    << "\n\t WTF = " << X
            //    << "\n\t MAYBE = " << Y
            //    << "\n";
            //}
            if (Src->getType() != DestTy) {
              auto *SrcFixed = CastInst::CreatePointerCast(Src, DestTy, "", SI);
              SI->getOperandUse(0).set(SrcFixed);
            }
          }
        } // end of handling store

        auto *PN = dyn_cast<PHINode>(&*I);
        if (N && PN) {
          if (!Targets.count(N)) {
            if (Refs.refersTargets(N)) {
              auto *NewType = getNewType(PN, NH, Refs);
              updateType(PN, NewType);
              for (unsigned i = 0; i < PN->getNumIncomingValues(); i++) {
                auto *Incoming = PN->getIncomingValue(i);
                if (isa<UndefValue>(Incoming)) {
                  updateType(Incoming, NewType);
                } else if (Incoming->getType() != NewType) {
                  auto *IncomingBB = PN->getIncomingBlock(i);
                  auto *IncomingFixed = CastInst::CreatePointerCast(
                      Incoming, NewType, "", IncomingBB->getTerminator());
                  PN->setIncomingValue(i, IncomingFixed);
                }
              }
            }
            continue;
          }
          DEBUG(errs() << "~~~~~ " << N << '\n');
          DEBUG(errs() << "--- handling phi\n");
          unsigned NumIncomings = PN->getNumIncomingValues();
          Instruction *InsertPt = &*I;
          auto *BasePHI =
              PHINode::Create(Int8PtrTy, NumIncomings, "base", InsertPt);
          auto *IdxPHI =
              PHINode::Create(Int64Ty, NumIncomings, "idx", InsertPt);
          auto *SizePHI =
              PHINode::Create(Int64Ty, NumIncomings, "size", InsertPt);

          DEBUG(errs() << "Setting triple for " << PN << '\n');
          TripleMap[PN] = {BasePHI, IdxPHI, SizePHI};

          std::vector<BasicBlock *> Predecessors;
          // add incoming values for PHIs of the triple
          for (unsigned i = 0; i < NumIncomings; i++) {
            Value *IncomingV = PN->getIncomingValue(i);
            BasicBlock *IncomingBB = PN->getIncomingBlock(i);
            Predecessors.push_back(IncomingBB);
            bool IsUndef = isa<UndefValue>(IncomingV);
            bool IsNull = isa<ConstantPointerNull>(IncomingV);

            assert(IsUndef || IsNull ||
                   Targets.count(getNodeForValue(IncomingV, F).getNode()) &&
                       "incoming value of a target-phi is not a target");

            const auto TripleIt = TripleMap.find(IncomingV);
            if (IsUndef) {
              BasePHI->addIncoming(UndefValue::get(Int8PtrTy), IncomingBB);
              IdxPHI->addIncoming(UndefValue::get(Int64Ty), IncomingBB);
              SizePHI->addIncoming(UndefValue::get(Int64Ty), IncomingBB);
            } else if (IsNull) {
              BasePHI->addIncoming(ConstantPointerNull::get(Int8PtrTy),
                                   IncomingBB);
              IdxPHI->addIncoming(UndefValue::get(Int64Ty), IncomingBB);
              SizePHI->addIncoming(UndefValue::get(Int64Ty), IncomingBB);
            } else if (TripleIt == TripleMap.end()) {
              // haven't computed triple for IncomingV, we will fix the PHIs
              // later
              IncomingValues[IncomingV].push_back(
                  {BasePHI, IdxPHI, SizePHI, IncomingBB});
            } else {
              const auto &Triple = TripleIt->second;
              BasePHI->addIncoming(Triple.Base, IncomingBB);
              IdxPHI->addIncoming(Triple.Idx, IncomingBB);
              SizePHI->addIncoming(Triple.Size, IncomingBB);
            }
          }

          for (auto *BB : Predecessors)
            PN->removeIncomingValue(BB, false /*delete if empty*/);

          removeValue(PN);
        } // end of handling phi

        auto *Cast = dyn_cast<CastInst>(&*I);
        if (Cast && Targets.count(N)) {
          DEBUG(errs() << "--- handling cast\n");
          auto *Src = Cast->getOperand(0);
          DEBUG(errs() << "Copying triple from " << Src << " to " << Cast
                       << '\n');
          assert(TripleMap.count(Src));
          assert(getNodeForValue(Src, F).getNode() == N);
          TripleMap[Cast] = TripleMap.at(Src);
          removeValue(Cast);
        } // end of handling bitcast

        if (auto CS = CallSite(&*I)) {
          DEBUG(errs() << "!!! handling call\n");
          // don't support indirect call
          if (!CS.getCalledFunction())
            continue;

          Function *Callee = CS.getCalledFunction();
          if (Callee->getIntrinsicID() == Intrinsic::lifetime_start ||
              Callee->getIntrinsicID() == Intrinsic::lifetime_end) {
            auto *Ptr = I->getOperand(1);
            if (Targets.count(getNodeForValue(Ptr, F).getNode()))
              I->getOperandUse(1).set(TripleMap.at(Ptr).Base);
          }
          // rewrite the allocation
          if (isAllocator(Callee)) {
            DEBUG(errs() << "ALLOCATOR: " << Callee->getName() << '\n');
            assert(CS.getNumArgOperands() == 1 ||
                   Callee->getName() == "calloc" ||
                   Callee->getName() == "realloc" && "Unknown allocator");
            auto *TotalSize = CS.getArgument(0);
            if (Callee->getName() == "calloc")
              TotalSize = BinaryOperator::CreateNSWMul(
                  TotalSize, CS.getArgument(1), "", &*I);
            else if (Callee->getName() == "realloc")
              TotalSize = CS.getArgument(1);

            if (Targets.count(N)) {
              DEBUG(errs() << "--- rewriting alloc\n");
              //
              // rewrite the allocation of type whose layout we changed
              //
              unsigned GroupId = Targets.lookup(N);
              Value *Size, *SizeInBytes, *Base;
              std::tie(Size, SizeInBytes) = computeAllocSize(
                  TotalSize, N, Transforms.at(GroupId).get(), &*I);
              if (Callee->getName() != "realloc")
                Base = rewriteAlloc(CS, SizeInBytes, &*I, "base");
              else {
                Value *OldPtr, *OldSize;
                const DSNode *OldN =
                    getNodeForValue(CS.getArgument(0), F).getNode();
                if (Targets.count(OldN)) {
                  assert(OldN == N);
                  auto OldTriple = TripleMap.at(CS.getArgument(0));
                  OldPtr = OldTriple.Base;
                  OldSize = OldTriple.Size;
                } else {
                  OldPtr = ConstantPointerNull::get(Int8PtrTy);
                  OldSize = ConstantInt::get(Int64Ty, 0);
                }
                auto *Realloc = getRealloc(Transforms.at(GroupId).get(), &M);
                Base = CallInst::Create(
                    Realloc, std::vector<Value *>{OldPtr, Size, OldSize},
                    I->getName(), &*I);
              }

              // fingers crossed an integer is either 64 bit or less
              if (Size->getType() != Int64Ty)
                Size = new ZExtInst(Size, Int64Ty, "size", &*I);

              DEBUG(errs() << "Setting triple for " << &*I << '\n');
              TripleMap[&*I] = {Base, Zero64, Size};
              removeValue(&*I);
            } else if (N && !N->isCollapsedNode() && Refs.refersTargets(N)) {
              //
              // rewrite the allocation of type that got triple-converted
              //
              DEBUG(errs() << "N = " << N
                           << ", ORIG N = " << Refs.getOrigNode(N) << '\n');
              unsigned SizeNeeded = computeSizeWithTripleConv(N, Refs);
              if (SizeNeeded != N->getSize()) {
                assert(SizeNeeded > N->getSize());
                auto *SizeTy = TotalSize->getType();
                Value *NumElems = BinaryOperator::CreateExactUDiv(
                    TotalSize,
                    ConstantInt::get(SizeTy, Refs.getOrigNode(N)->getSize()),
                    "num_elems", &*I);
                Value *SizeToAlloc = BinaryOperator::CreateNSWMul(
                    NumElems, ConstantInt::get(SizeTy, SizeNeeded), "new_size",
                    &*I);
                auto *NewAlloc =
                    rewriteAlloc(CS, SizeToAlloc, &*I, I->getName());
                I->replaceAllUsesWith(NewAlloc);
                removeValue(&*I);
              }
            }
            continue;
          } else if (isDeallocator(Callee)) {
            auto *Ptr = CS.getArgument(0);
            if (Targets.count(getNodeForValue(Ptr, F).getNode())) {
              DEBUG(errs() << "--- rewriting dealloc\n");
              assert(CS.getNumArgOperands() == 1 &&
                     "What kind of deallocator is this???");
              assert(TripleMap.count(Ptr) &&
                     "Triple not found for target to be decallocated");
              std::vector<Value *> DeallocArgs = {TripleMap.at(Ptr).Base};
              if (auto *Invoke = dyn_cast<InvokeInst>(CS.getInstruction()))
                InvokeInst::Create(Callee, Invoke->getNormalDest(),
                                   Invoke->getUnwindDest(), DeallocArgs, "",
                                   &*I);
              else
                CallInst::Create(Callee, DeallocArgs, "", &*I);
              removeValue(&*I);
            }
            continue;
          } // end of handling allocation

          if (Callee->getIntrinsicID() == Intrinsic::memcpy) {
            DSNodeHandle NH = getNodeForValue(CS.getArgument(0), F),
                         MH = getNodeForValue(CS.getArgument(1), F);
            if (!Refs.refersTargets(NH.getNode()))
              continue;
            assert(NH.getNode() == MH.getNode());
            assert(NH.getOffset() == MH.getOffset());
            assert(Offsets.lookup(NH.getNode()) + NH.getOffset() == 0);
            assert(cast<ConstantInt>(CS.getArgument(2))->getZExtValue() ==
                   Refs.getOrigNode(NH.getNode())->getSize());
            auto *NewSize =
                ConstantInt::get(CS.getArgument(2)->getType(),
                                 computeSizeWithTripleConv(NH.getNode(), Refs));
            I->getOperandUse(2).set(NewSize);
          }

          if (Callee->empty())
            continue;

          DEBUG(errs() << "--- handling function call to " << Callee->getName()
                       << '\n');

          auto *CallerG = getDSGraph(F), *CalleeG = getDSGraph(Callee);
          // propagate targets from caller to callee
          // also propagate the group tag
          TargetMapTy TargetsInCallee;
          RefSetTracker<TargetMapTy> CalleeRefs;
          OffsetMapTy CalleeOffsets;
          bool SameSCC = (DSA->getCallGraph().sccLeader(getSrcFunc(F)) ==
                          DSA->getCallGraph().sccLeader(getSrcFunc(Callee)));
          DEBUG(errs() << F->getName() << " and " << Callee->getName()
                       << " in the same SCC: " << SameSCC << '\n');
          const auto &CalleeCallerMapping =
              getCalleeCallerMapping({CS, Callee});
          for (auto Mapping : CalleeCallerMapping) {
            auto *CalleeN = Mapping.first;
            auto *CallerN = Mapping.second.getNode();
            auto It = Targets.find(CallerN);
            if (It != Targets.end()) {
              unsigned GroupId = It->second;
              TargetsInCallee[CalleeN] = GroupId;
              CalleeOffsets[CalleeN] =
                  Offsets.lookup(CallerN) + Mapping.second.getOffset();
              assert(Offsets.lookup(CallerN) + Mapping.second.getOffset() <
                     Transforms.at(GroupId)->getSize());
            }
          }

          CloneSignature Sig(TargetsInCallee, CalleeOffsets,
                             getSrcFunc(Callee));
          for (auto TargetAndGroup : TransTargets) {
            const DSNode *N = TargetAndGroup.first;
            if (N->getParentGraph() == CalleeG && !TargetsInCallee.count(N)) {
              TargetsInCallee[N] = TargetAndGroup.second;
              CalleeOffsets[N] = 0;
            }
          }
          CalleeRefs =
              Refs.getCalleeTracker(TargetsInCallee, CalleeCallerMapping);

          if (shouldCloneFunction(Callee, CalleeRefs)) {
            DEBUG(errs() << "--- need to clone callee for call"
                << *I
                << "\n");
            auto *CloneFnTy = computeCloneFnTy(Callee, CalleeRefs);
            DEBUG(errs() << "--- inferred new callee type after cloning: "
                         << *CloneFnTy << '\n');

            Function *Clone;
            // maybe we have a clone for this use already
            const auto CloneIt = CloneCache.find(Sig);
            if (CloneIt != CloneCache.end())
              Clone = CloneIt->second;
            else {
              DEBUG(errs() << "--- cloning " << Callee->getName() << '\n');
              Clone = cloneFunction(Callee, CloneFnTy, TargetsInCallee,
                                    TripleMap, Sig, Targets.count(N));
              DEBUG(errs() << "-- cloned function: " << Clone->getName()
                           << '\n');
              // we haven't processed this new clone
              Worklist.push_back({Clone, CalleeRefs, CalleeOffsets});
            }

            //
            // compute the list of arguments with which we call the clone
            //
            std::vector<Value *> NewArgs;
            //
            // if the original function call returns pointer to Target
            // we need to return by reference
            //
            AllocaInst *BaseAddr, *IdxAddr, *SizeAddr;
            bool ReturnsTarget = Targets.count(
                getNodeForValue(CS.getInstruction(), F).getNode());
            unsigned i = 0;
            if (ReturnsTarget) {
              // allocate 3 slots on stack for the callee to set return value of
              // the triple
              Instruction *Begin = &*F->getEntryBlock().getFirstInsertionPt();
              BaseAddr = new AllocaInst(Int8PtrTy, "base.ret", Begin);
              IdxAddr = new AllocaInst(Int64Ty, "idx.ret", Begin);
              SizeAddr = new AllocaInst(Int64Ty, "size.ret", Begin);
              NewArgs.push_back(BaseAddr);
              NewArgs.push_back(IdxAddr);
              NewArgs.push_back(SizeAddr);
              i = 3;
            }
            for (Value *Arg : CS.args()) {
              if (Targets.count(getNodeForValue(Arg, F).getNode())) {
                DEBUG(errs() << "Looking up triple for " << *Arg << "\n");
                auto &Triple = TripleMap.at(Arg);
                assert(Triple.Base);
                NewArgs.push_back(Triple.Base);
                NewArgs.push_back(Triple.Idx);
                NewArgs.push_back(Triple.Size);
                i += 3;
              } else {
                if (i < CloneFnTy->params().size()) {
                  auto *ParamTy = CloneFnTy->params()[i++];
                  if (Arg->getType() != ParamTy)
                    Arg = CastInst::CreatePointerCast(Arg, ParamTy,
                                                      Arg->getName(), &*I);
                }
                NewArgs.push_back(Arg);
              }
            }

            //
            // call the clone
            //
            Value *NewCall;
            if (auto *Invoke = dyn_cast<InvokeInst>(&*I))
              NewCall =
                  InvokeInst::Create(Clone, Invoke->getNormalDest(),
                                     Invoke->getUnwindDest(), NewArgs, "", &*I);
            else
              NewCall = CallInst::Create(Clone, NewArgs, "", &*I);

            //
            // use the call to the clone instead of the old one
            //
            if (!ReturnsTarget) {
              updateType(&*I, NewCall->getType());
              I->replaceAllUsesWith(NewCall);
              removeValue(&*I);
            } else {
              //
              // load the triple being returned
              //
              Instruction *InsertPt;
              if (auto *Invoke = dyn_cast<InvokeInst>(&*I))
                InsertPt = &*Invoke->getNormalDest()->getFirstInsertionPt();
              else
                InsertPt = &*std::next(I);
              auto *Base = new LoadInst(BaseAddr, "base", InsertPt);
              auto *Idx = new LoadInst(IdxAddr, "idx", InsertPt);
              auto *Size = new LoadInst(SizeAddr, "size", InsertPt);
              DEBUG(errs() << "Setting triple for " << &*I << '\n');
              TripleMap[&*I] = {Base, Idx, Size};
              removeValue(&*I);
            }
            updateValue(&*I, NewCall);
          }
        } // end of handling function call

        //
        if (auto *RI = dyn_cast<ReturnInst>(&*I)) {
          DEBUG(errs() << "-- handling return\n");

          //
          // if a pointer to the target is being returned, the function will be
          // transformed to return by reference, so we need to replace the
          // return instruction with a store to triple (last 3 elements of the
          // argument list)
          //
          if (F->getReturnType() == VoidTy) {
            auto *RetVal = RI->getReturnValue();
            if (!RetVal || !Targets.count(getNodeForValue(RetVal, F).getNode()))
              continue;

            assert(TripleMap.count(RetVal));
            auto &Triple = TripleMap.at(RetVal);
            auto BaseAddr = F->getArgumentList().begin(),
                 IdxAddr = std::next(BaseAddr), SizeAddr = std::next(IdxAddr);

            new StoreInst(
                CastInst::CreatePointerCast(Triple.Base, Int8PtrTy, "", RI),
                &*BaseAddr, RI);
            new StoreInst(Triple.Idx, &*IdxAddr, RI);
            new StoreInst(Triple.Size, &*SizeAddr, RI);
            ReturnInst::Create(RI->getContext(), nullptr, RI);
            RI->eraseFromParent();
          } else if (RI->getReturnValue()->getType() != F->getReturnType()) {
            // in case we changed the return type
            auto *OldRetVal = RI->getReturnValue();
            auto *RetVal = CastInst::CreatePointerCast(
                OldRetVal, F->getReturnType(), OldRetVal->getName(), RI);
            ReturnInst::Create(RetVal->getContext(), RetVal, RI);
            removeValue(RI);
          }
        } // end of handling return

        auto *AI = dyn_cast<AllocaInst>(&*I);
        if (AI && N) {
          DEBUG(errs() << "--- handling alloca\n");
          //
          // We need to fix this allocation if it's a target (change size) or it
          // points to a target (pointer conv.)
          //

          if (Targets.count(N)) {
            unsigned GroupId = Targets.lookup(N);
            // figure out total bytes allocated in the original allocation
            //
            // FIXME: this is not correct
            // consider `alloca i32, 32` vs `alloca [32 x i32]`
            Type *SizeTy = AI->getArraySize()->getType();
            auto *OrigElemSize = ConstantInt::get(
                SizeTy, TD->getTypeAllocSize(AI->getAllocatedType()));
            auto *OrigBytesAllocated = BinaryOperator::CreateNSWMul(
                OrigElemSize, AI->getArraySize(), "", AI);

            // figure out bytes needed after allocation
            Value *Size, *SizeInBytes;
            std::tie(Size, SizeInBytes) = computeAllocSize(
                OrigBytesAllocated, N, Transforms.at(GroupId).get(), AI);
            auto *NewAlloc =
                new AllocaInst(Int8Ty, SizeInBytes, AI->getName(), AI);
            updateValue(AI, NewAlloc);
            removeValue(AI);
            TripleMap[AI] = {NewAlloc, Zero64, Size};
            continue;
          }

          //
          // rewrite the allocation if the object points to a target
          // and needs to be triple-converted
          //

          if (!Refs.refersTargets(N))
            continue;

          unsigned SizeNeeded = computeSizeWithTripleConv(N, Refs);

          auto *TyAfterConv =
              cast<PointerType>(getNewType(AI, NH, Refs))->getElementType();
          // auto *TyAfterConv = getNewType(NH.getNode(), Refs,
          // AI->getAllocatedType(), NH.getOffset());
          // sometimes the alloca is not typed, so we need to check both the
          // type and size
          if (SizeNeeded > N->getSize() || TyAfterConv != AI->getType()) {
            AllocaInst *NewAlloca;
            Type *AllocatedType = AI->getAllocatedType(),
                 *NewAllocatedType = TyAfterConv;
            if (auto *AT = dyn_cast<ArrayType>(AllocatedType)) {
              AllocatedType = AT->getElementType();
              NewAllocatedType = cast<ArrayType>(TyAfterConv)->getElementType();
            }
            //
            // There are two cases here.
            // case 1: The allocation is typed. e.g. `i32* array = alloca i32,
            // i64 n`
            // case 2: The allocation is untyped, like malloc. e.g. `i32* array
            // = alloca i8, i64 size_in_bytes`.
            //
            if (TD->getTypeAllocSize(AllocatedType) == N->getSize() &&
                TD->getTypeAllocSize(NewAllocatedType) >= SizeNeeded) {
              // case 1
              // this is the easy one, we just change the type being allocated
              NewAlloca =
                  new AllocaInst(TyAfterConv, AI->getArraySize(), "", AI);
            } else {
              DEBUG(errs() << "OLD ALLOCA " << *AI << "\n\t, OLD TYPE"
                           << *AllocatedType << ", NEW TYPE"
                           << *NewAllocatedType << ", OLD SIZE " << N->getSize()
                           << ", NEW SIZE "
                           << TD->getTypeAllocSize(NewAllocatedType) << '\n');
              // case 2
              // need to calculate number of bytes needed
              assert(AI->getAllocatedType() == Int8Ty &&
                     "unsupported pattern of alloca");
              auto *OrigNumBytes = AI->getArraySize();
              auto *SizeTy = OrigNumBytes->getType();
              auto *OldElemSize = ConstantInt::get(SizeTy, N->getSize()),
                   *ElemSize = ConstantInt::get(SizeTy, SizeNeeded);
              auto *ArrSize = BinaryOperator::CreateExactUDiv(
                  OrigNumBytes, OldElemSize, "arr_size", AI);
              auto *NumBytes = BinaryOperator::CreateNSWMul(ArrSize, ElemSize,
                                                            "new_size", AI);
              NewAlloca = new AllocaInst(Int8Ty, NumBytes);
            }
            auto *OldTy = AI->getType();
            AI->mutateType(NewAlloca->getType());
            AI->replaceAllUsesWith(NewAlloca);
            AI->mutateType(OldTy);
            DEBUG(errs() << "REPLACED " << *AI << '\n');
            updateValue(AI, NewAlloca);
            removeValue(AI);
          }
        } // end of handling alloca

        if (auto *ICMP = dyn_cast<ICmpInst>(&*I)) {
          DEBUG(errs() << "--- handling icmp\n");
          auto *Ty = ICMP->getOperand(0)->getType();

          for (unsigned i = 0, e = ICMP->getNumOperands(); i != e; i++) {
            auto *OrigOp = ICMP->getOperand(i);
            auto NH = getNodeForValue(OrigOp, F);
            auto *N = NH.getNode();
            if (Targets.count(N)) {
              unsigned GroupId = Targets.lookup(N);
              unsigned Offset = Offsets.lookup(N) + NH.getOffset();
              assert(TripleMap.count(OrigOp));
              auto &Triple = TripleMap.at(OrigOp);
              assert(Transforms.count(GroupId));
              auto *Addr = Transforms.at(GroupId)->ComputeAddress(
                  Triple, Offset, ICMP, true /* don't care */);
              auto *Ptr = CastInst::CreatePointerCast(Addr, Ty,
                                                      OrigOp->getName(), ICMP);
              ICMP->getOperandUse(i).set(Ptr);
            } else if (OrigOp->getType() != Ty) {
              auto *FixedOp = CastInst::CreatePointerCast(
                  OrigOp, Ty, OrigOp->getName(), ICMP);
              ICMP->getOperandUse(i).set(FixedOp);
            }
          }
        } // end of handling icmp
      }
    }

    for (const auto &Pair : IncomingValues) {
      auto &Triple = TripleMap.at(Pair.first);
      for (auto &IncomingTriple : Pair.second) {
        IncomingTriple.IncomingBase->addIncoming(Triple.Base,
                                                 IncomingTriple.IncomingBB);
        IncomingTriple.IncomingIdx->addIncoming(Triple.Idx,
                                                IncomingTriple.IncomingBB);
        IncomingTriple.IncomingSize->addIncoming(Triple.Size,
                                                 IncomingTriple.IncomingBB);
      }
    }
  }
}

FunctionType *
LayoutTuner::computeCloneFnTy(Function *F,
                              const RefSetTracker<TargetMapTy> &Refs) {
  DEBUG(errs() << "----- computing clone fn ty\n");
  DEBUG(errs() << "---- getting function type for function " << F->getName()
               << '\n');
  auto *OrigTy = F->getFunctionType();
  std::vector<Type *> ArgTypes;

  DSNodeHandle RetNH = getDSGraph(F)->getReturnNodeFor(*F);

  DEBUG(errs() << "---- deducing return type for clone\n");
  Type *RetTy;
  if (Refs.getTargets().count(RetNH.getNode())) {
    // in case the function returns a pointer to target
    // we append 3 extra return arguments to set the triple begin "returned"
    RetTy = VoidTy;
    ArgTypes.push_back(PointerType::getUnqual(Int8PtrTy));
    ArgTypes.push_back(PointerType::getUnqual(Int64Ty));
    ArgTypes.push_back(PointerType::getUnqual(Int64Ty));
  } else if (OrigTy->getReturnType() != VoidTy) {
    RetTy = getNewType(RetNH.getNode(), Refs, OrigTy->getReturnType(),
                       RetNH.getOffset());
  } else
    RetTy = VoidTy;

  DEBUG(errs() << "--- deducing input args for clone\n");
  for (auto &Arg : F->args()) {
    DSNodeHandle ArgNH = getNodeForValue(&Arg, F);
    auto *ArgN = ArgNH.getNode();

    if (Refs.getTargets().count(ArgN)) {
      // expand the pointer to target into triple
      ArgTypes.push_back(Int8PtrTy);
      ArgTypes.push_back(Int64Ty);
      ArgTypes.push_back(Int64Ty);
    } else {
      if (!ArgN) {
        ArgTypes.push_back(Arg.getType());
        continue;
      }

      bool HasRef = false;
      for (unsigned Offset = ArgNH.getOffset(); Offset < ArgN->getSize();
           Offset++)
        if (Refs.hasRefAt(ArgN, Offset)) {
          HasRef = true;
          break;
        }
      if (HasRef) {
        ArgTypes.push_back(
            getNewType(const_cast<Value *>(getOrigValue(&Arg)), ArgN, Refs));
      } else {
        ArgTypes.push_back(Arg.getType());
      }
    }
  }

  return FunctionType::get(RetTy, ArgTypes, F->getFunctionType()->isVarArg());
}

bool LayoutTuner::shouldCloneFunction(
    Function *F, const RefSetTracker<TargetMapTy> &Refs) const {
  // find functions where we begin the transformation
  // these are the functions that we don't clone
  // we want function whose arguments from which the node we are transforming is
  // not reachable
  llvm::DenseSet<const DSNode *> ReachableNodes;
  for (auto &Arg : F->args()) {
    auto NH = getNodeForValue(&Arg, F);
    if (!NH.isNull())
      NH.getNode()->markReachableNodes(ReachableNodes);
  }

  // also if the function returns a pointer to target
  if (auto *RetNode = getDSGraph(F)->getReturnNodeFor(*F).getNode()) {
    RetNode->markReachableNodes(ReachableNodes);
  }

  for (auto *N : ReachableNodes) {
    if (Refs.getTargets().count(N) || Refs.refersTargets(N))
      return true;
  }

  return false;
}

// FIXME: need to propagate Refs so that we can make sure indirect calls don't
// touch refs as well (since these refs's layout will be changed as well)
TargetMapTy LayoutTuner::findLegalTargets(const Module &M) {
  std::vector<const DSNode *> Candidates;
  auto &TypeChecker = getAnalysis<TypeSafety<EquivBUDataStructures>>();
  // mapping node -> set of unique targets it correspond to
  typedef std::map<const DSNode *, std::set<unsigned>> TargetTrackingMap;
  std::map<DSGraph *, std::map<const DSNode *, std::set<unsigned>>> TargetMaps;
  std::map<DSGraph *, std::set<const DSNode *>> LocalNodes;
  std::map<const DSNode *, std::set<unsigned>> NodesWithZeroOffset;
  std::set<unsigned> Disqualified;

  auto IsLegalTarget = [this, &TypeChecker](const DSNode *N) -> bool {
    if (N->isGlobalNode() || N->getSize() < MinSize ||
       (ArrayOnly && !N->isArrayNode()) ||
       !TypeChecker.isTypeSafe(N))
      return false;

    unsigned BytesUsed = 0;
    // don't mess with fields that are composite types or union
    for (auto ti = N->type_begin(), te = N->type_end(); ti != te; ++ti) {
      if (ti->second->size() != 1)
        return false;
      for (Type *Ty : *ti->second)
        if (isa<CompositeType>(Ty))
          return false;
        else if (NoPointer && Ty->isPointerTy())
          return false;
        else if (NoFP)
          if (auto *Pointer = dyn_cast<PointerType>(Ty))
            if (Pointer->getElementType()->isFunctionTy())
              return false;

      BytesUsed += TD->getTypeAllocSize(*ti->second->begin());
    }

    return std::distance(N->type_begin(), N->type_end()) >= MinFields;
  };

  auto FindTargetsReferred =
      [&TargetMaps](const DSNode *N, std::set<unsigned> &TargetsReferred) {
        auto &Targets = TargetMaps[N->getParentGraph()];
        for (auto ei = N->edge_begin(), ee = N->edge_end(); ei != ee; ++ei) {
          auto It = Targets.find(ei->second.getNode());
          if (It != Targets.end())
            insertInto(TargetsReferred, It->second);
        }
      };

  // find typesafe local nodes in each graph
  for (auto *G : SortedSCCs)
    for (auto ni = G->node_begin(), ne = G->node_end(); ni != ne; ++ni) {
      DSNode *N = &*ni;
      if (IsLegalTarget(N))
        LocalNodes[G].insert(N);
    }

  bool Changed;
  bool FirstPass = true;

  auto *GG = DSA->getGlobalsGraph();
  auto &GlobalTargets = TargetMaps[GG];
  for (auto ni = GG->node_begin(), ne = GG->node_end(); ni != ne; ++ni) {
    DSNode *N = &*ni;
    if (IsLegalTarget(N) && !refersTargets(N, NodeSetTy{N}) &&
        !refersTargets(N, GlobalTargets)) {
      unsigned TargetId = Candidates.size();
      Candidates.push_back(N);
      GlobalTargets[N].insert(TargetId);
    }
  }

  //
  // a) find out all unique type-safe targets n s.t. given
  //    a corresponding node m in a DSGraph n's offset
  //    from m is constant regardless calling context
  // b) find out which nodes has zero offset with a target
  //
  do {
    Changed = false;
    for (auto *G : SortedSCCs) {
      auto &Targets = TargetMaps[G];

      if (FirstPass) {
        // propagate nodes from global graph
        auto &GGMapping = getGToGGMapping(G);
        for (auto &Mapping : GGMapping) {
          auto &GGNH = Mapping.second;
          auto It = GlobalTargets.find(GGNH.getNode());
          if (It != GlobalTargets.end())
            insertInto(Targets[Mapping.first], It->second);
        }

        for (auto *N : LocalNodes[G])
          // found new unique target
          if (!Targets.count(N)) {
            unsigned TargetId = Candidates.size();
            Candidates.push_back(N);
            Targets[N].insert(TargetId);
            NodesWithZeroOffset[N].insert(TargetId);
            Changed = true;
          }
      }

      // remove nodes that refer to other nodes
      for (const auto &Pair : Targets) {
        const DSNode *N = Pair.first;
        auto &TargetIds = Pair.second;
        for (auto ei = N->edge_begin(), ee = N->edge_end(); ei != ee; ++ei) {
          auto It = Targets.find(ei->second.getNode());
          if (It == Targets.end())
            continue;
          bool RefersTargets = false;
          for (unsigned ReferredTargetId : It->second)
            if (!Disqualified.count(ReferredTargetId)) {
              insertInto(Disqualified, TargetIds);
              RefersTargets = true;
              break;
            }
          if (RefersTargets)
            break;
        }
      }

      std::map<std::pair<const DSNode *, const DSNode *>, unsigned> RelOffsets;

      // propagate targets to successor
      // during propagation we also check that for every mapping from
      // DSNode v to DSNode w, there's only one unique offset.
      // If not, targets correspond to v are disqualified.
      for (auto Edge : OutgoingEdges[G]) {
        const auto &CalleeCallerMapping = getCalleeCallerMapping(Edge);
        auto *CalleeG = DSA->getDSGraph(*Edge.second);
        for (auto Mapping : CalleeCallerMapping) {
          const DSNode *CalleeN;
          DSNodeHandle CallerNH;
          std::tie(CalleeN, CallerNH) = Mapping;
          if (!Targets.count(CallerNH.getNode()))
            continue;

          auto &TargetIds = Targets.at(CallerNH.getNode());
          // auto OffsetIt = RelOffsets.find({CallerNH.getNode(), CalleeN});
          // if (false && OffsetIt != RelOffsets.end() &&
          //    OffsetIt->second != CallerNH.getOffset()) {
          //  // we've already found this mapping
          //  // check that their's only one relative offset
          //  insertInto(Disqualified, TargetIds);
          //}

          RelOffsets[{CallerNH.getNode(), CalleeN}] = CallerNH.getOffset();

          auto &CalleeTargetIds = TargetMaps[CalleeG][CalleeN];
          unsigned OldSize = CalleeTargetIds.size();
          insertInto(CalleeTargetIds, TargetIds);
          Changed |= (CalleeTargetIds.size() != OldSize);

          if (CallerNH.getOffset() == 0)
            NodesWithZeroOffset[CalleeN].insert(
                NodesWithZeroOffset[CallerNH.getNode()].begin(),
                NodesWithZeroOffset[CallerNH.getNode()].end());
        }
      }
    }

    FirstPass = false;
  } while (Changed);

  RefSetTracker<decltype(GlobalTargets)> GGRefs(GlobalTargets);

  // There are certain patterns of global pointer to targets that we support
  // filter out those that aren't. These cases are rare and not worth supporting
  // for now...
  for (auto &G : M.globals()) {
    auto NH = GG->getNodeForValue(&G);
    auto *N = NH.getNode();
    if (!N)
      continue;

    std::set<unsigned> GlobalsReferred;
    FindTargetsReferred(N, GlobalsReferred);

    if (!GlobalsReferred.empty() && !N->isCollapsedNode()) {
      // for now, to simplify transformation,
      // we support handling global that refers to target
      // only if the global is declared with type infered by DSA
      unsigned SizeNeeded = computeSizeWithTripleConv(N, GGRefs);

      auto *TyAfterConv =
          cast<PointerType>(
              getNewType(const_cast<GlobalVariable *>(&G), NH, GGRefs))
              ->getElementType();
      if (true || TD->getTypeAllocSize(TyAfterConv) != SizeNeeded) {
        insertInto(Disqualified, GlobalsReferred);
        continue;
      }

      // for now only allow such globals to be used in instructions directly
      for (auto *U : G.users())
        if (isa<Constant>(U)) {
          insertInto(Disqualified, GlobalsReferred);
          continue;
        }
    } else if (N->isCollapsedNode())
      insertInto(Disqualified, GlobalsReferred);
  }

  // 1) make sure that a gep either
  //    a) index with a constant offset
  // or b) index with a src type with the same size as the target's
  // 2) make sure we no indirect accesses target
  //
  for (auto &F : M) {
    if (F.empty())
      continue;

    auto *G = getDSGraph(&F);
    auto &Targets = TargetMaps[G];
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          auto SrcNH = getNodeForValue(GEP->getPointerOperand(), &F);
          if (!Targets.count(SrcNH.getNode()))
            continue;

          auto *SrcN = SrcNH.getNode();
          unsigned SrcSize = TD->getTypeAllocSize(GEP->getSourceElementType());

          // if SrcN correspond to a target and this gep doesn't process
          // a constant offset, this better have identical size as SrcSize
          // offset at 0 with SrcN
          std::set<unsigned> CantChangeIdx;
          for (unsigned TargetId : Targets.at(SrcN)) {
            if (Disqualified.count(TargetId))
              continue;
            const DSNode *Target = Candidates[TargetId];
            if (Target->getSize() != SrcSize ||
                !NodesWithZeroOffset[SrcN].count(TargetId))
              CantChangeIdx.insert(TargetId);
          }

          APInt RelOffset(TD->getPointerSizeInBits(GEP->getAddressSpace()), 0);
          bool HasConstOffset = GEP->accumulateConstantOffset(*TD, RelOffset);
          if (!HasConstOffset) {
            insertInto(Disqualified, CantChangeIdx);
          }
        } else if (auto CS = CallSite(const_cast<Instruction *>(&I))) {
          //
          // make sure no indirect call can access any target
          //
          if (!CS.getCalledFunction()) {
            DenseSet<const DSNode *> ReachableNodes;
            auto *RetN = G->getNodeForValue(CS.getInstruction()).getNode();
            if (RetN)
              RetN->markReachableNodes(ReachableNodes);

            for (auto &Arg : CS.args()) {
              auto *N = G->getNodeForValue(Arg.get()).getNode();
              if (!N)
                continue;
              N->markReachableNodes(ReachableNodes);
            }

            for (auto *N : ReachableNodes) {
              auto It = Targets.find(N);
              if (It != Targets.end())
                insertInto(Disqualified, It->second);
              for (auto ei = N->edge_begin(), ee = N->edge_end(); ei != ee;
                   ++ei) {
                const DSNode *M = ei->second.getNode();
                auto It = Targets.find(M);
                if (It != Targets.end())
                  insertInto(Disqualified, It->second);
              }
            }
          }

          //
          // make sure we don't pass a target to memmove, and memset
          //
          std::set<unsigned> TargetsUsed;
          auto It = Targets.find(G->getNodeForValue(&I).getNode());
          if (It != Targets.end())
            insertInto(TargetsUsed, It->second);

          for (auto &Arg : CS.args()) {
            auto *N = G->getNodeForValue(Arg.get()).getNode();
            if (!N)
              continue;
            auto It = Targets.find(N);
            if (It != Targets.end())
              insertInto(TargetsUsed, It->second);
            FindTargetsReferred(N, TargetsUsed);
          }

          if (TargetsUsed.empty())
            continue;

          auto *Callee = CS.getCalledFunction();
          bool Supported =
              Callee && (!Callee->isDeclarationForLinker() || isDeallocator(Callee) ||
              isAllocator(Callee) ||
              Callee->getIntrinsicID() == Intrinsic::lifetime_start ||
              Callee->getIntrinsicID() == Intrinsic::lifetime_end ||
              Callee->getIntrinsicID() == Intrinsic::memcpy);
          if (!Supported) {
            insertInto(Disqualified, TargetsUsed);
          } else if (Callee->getIntrinsicID() == Intrinsic::memcpy) {
            const DSNode *N = G->getNodeForValue(CS.getArgument(0)).getNode();
            if (!refersTargets(N, Targets) && !Targets.count(N))
              continue;
            if (Targets.count(N))
              insertInto(Disqualified, Targets.at(N));
            else if (!isa<ConstantInt>(CS.getArgument(2)))
              FindTargetsReferred(N, Disqualified);
          } else if (Callee->getName() == "realloc") {
            const DSNode *N = G->getNodeForValue(CS.getArgument(0)).getNode();
            if (!Targets.count(N))
              insertInto(Disqualified, TargetsUsed);
          }
        } else if (auto *IV = dyn_cast<InsertValueInst>(&I)) {
          auto *N = G->getNodeForValue(IV->getAggregateOperand()).getNode();
          auto It = Targets.find(N);
          if (It != Targets.end())
            insertInto(Disqualified, It->second);

          N = G->getNodeForValue(IV->getInsertedValueOperand()).getNode();
          It = Targets.find(N);
          if (It != Targets.end())
            insertInto(Disqualified, It->second);
        } else if (auto *EV = dyn_cast<ExtractValueInst>(&I)) {
          auto *N = G->getNodeForValue(EV->getAggregateOperand()).getNode();
          auto It = Targets.find(N);
          if (It != Targets.end())
            insertInto(Disqualified, It->second);

          N = G->getNodeForValue(EV).getNode();
          It = Targets.find(N);
          if (It != Targets.end())
            insertInto(Disqualified, It->second);
        } else if (auto *ITP = dyn_cast<IntToPtrInst>(&I)) {
          auto *N = G->getNodeForValue(ITP).getNode();
          auto It = Targets.find(N);
          if (It != Targets.end())
            insertInto(Disqualified, It->second);
          if (N)
            FindTargetsReferred(N, Disqualified);
        } else if (auto *PTI = dyn_cast<PtrToIntInst>(&I)) {
          auto *N = G->getNodeForValue(PTI->getPointerOperand()).getNode();
          auto It = Targets.find(N);
          if (It != Targets.end())
            insertInto(Disqualified, It->second);
        }
  }

  // now filter out nodes referred by type-unsafe nodes
  for (auto &GraphAndTargets : TargetMaps) {
    const DSGraph *G = GraphAndTargets.first;
    auto &Targets = GraphAndTargets.second;
    for (auto ni = G->node_begin(), ne = G->node_end(); ni != ne; ++ni) {
      const DSNode *N = &*ni;
      if (N->isIntToPtrNode() || !TypeChecker.isTypeSafeIfInternal(N)) {
        unsigned old = Disqualified.size();
        FindTargetsReferred(N, Disqualified);
      }
    }
  }

  errs() << "NUM DISQUALIFIED " << Disqualified.size() << '\n';

  TargetMapTy LegalTargets;
  for (unsigned i = 0, e = Candidates.size(); i != e; i++)
    if (!Disqualified.count(i)) {
      auto *N = Candidates[i];
      unsigned GroupId = LegalTargets.size();
      if (N->getParentGraph() == GG)
        DEBUG(errs() << "GROUP " << GroupId << " is in globals graph\n");
      else
        DEBUG(errs() << "GROUP " << GroupId << " is in SCC of "
                     << N->getParentGraph()->retnodes_begin()->first->getName()
                     << "(G= " << N->getParentGraph() << ")\n");
      LegalTargets[Candidates[i]] = GroupId;
    }

  return LegalTargets;
}

int main(int argc, char **argv) {
  std::srand(42);
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "tuning data layout of a program");

  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);
  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  errs() << "OUT = " << OutputFilename << '\n';

  legacy::PassManager Passes;
  Passes.add(new LayoutTuner());
  Passes.add(createVerifierPass());
  Passes.run(*M);

  std::error_code EC;
  tool_output_file Out(OutputFilename, EC, sys::fs::F_None);
  if (EC) {
    errs() << EC.message() << '\n';
    return 1;
  }

  WriteBitcodeToFile(M.get(), Out.os());

  Out.keep();
}
