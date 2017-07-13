#include "dsa/DSGraph.h"
#include "dsa/DataStructure.h"
#include "dsa/InitializeDSAPasses.h"
#include "dsa/TypeSafety.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Bitcode/BitcodeWriterPass.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GetElementPtrTypeIterator.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Pass.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/SystemUtils.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include "Transform.h"

//
// ========== RAAAAAAAAAAAAAAAAAAAAAAAAAMBLING ==========
// 1) Maybe we should treat nodes that refers to target as the same way we treat targets?
// perhaps with something TargetRefs? The primary rationale here is so that we
// can track the actual size of these nodes acurately
//
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
// 3) think of a way to do legality check in a single pass
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

struct IndexTriple {
  Value *Base, *Idx, *Size;
};

typedef std::vector<Value *> GEPIdxTy;

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
protected:
  StructType *TargetTy;
  Constant *ElemSize;
  std::map<unsigned, unsigned> Offset2IdxMap;

  TransformLowering(const DSNode *Target);

public:
  Constant *getOrigSize() const { return ElemSize; }

  // compute the address of an element after transformation
  virtual Value *ComputeAddress(IndexTriple Triple, unsigned Offset,
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

// a set of target nodes tagged with its group idx
typedef std::map<const DSNode *, unsigned> TargetMapTy;

class TargetSetTy {
  std::vector<std::pair<const DSNode *, unsigned>> Targets;

public:
  TargetSetTy(TargetMapTy TheTargets) {
    for (auto TargetAndGroup : TheTargets)
      Targets.push_back(TargetAndGroup);
  }

  bool operator<(const TargetSetTy Other) const {
    return Targets < Other.Targets;
  }
};

class LayoutTuner : public ModulePass {
  typedef std::set<const DSNode *> NodeSetTy;
  typedef std::pair<TargetSetTy, Function *> CloneSignature;
  // edge between two SCCs
  typedef std::pair<CallSite, const Function *> CallEdgeTy;
  // mapping a DSNode's offset from a target node
  typedef std::map<const DSNode *, std::set<unsigned>> OffsetMapTy;

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
    RefRecord(const DSNode *N, const NODE_SET_TY &Targets) {
      for (auto fi = N->type_begin(), fe = N->type_end(); fi != fe; ++fi) {
        unsigned Offset = fi->first;
        if (N->hasLink(Offset) && Targets.count(N->getLink(Offset).getNode()))
          References.insert(Offset);
      }
    }

    //
    // remember the fact that CallerNH points to a set of targets
    // in the context of Callee
    //
    RefRecord(DSNodeHandle CallerNH, const DSNode *CalleeN,
              const RefRecord &CallerRefs) {
      unsigned Offset = CallerNH.getOffset();
      for (unsigned RefOffset : CallerRefs.References) {
        if (RefOffset < Offset)
          continue;
        if (RefOffset - Offset < CalleeN->getSize())
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
          errs() << "Caller size: " << CallerN->getSize() << '\n';
          errs() << "Callee size: " << CalleeN->getSize() << '\n';
          errs() << "Callee offset: " << CallerNH.getOffset() << '\n';
          CalleeTracker.RefRecords[CalleeN] = RefRecord(CallerNH, CalleeN, Ref);
          errs() << "CALLEE HAS REF? "
                 << CalleeTracker.RefRecords.at(CalleeN).hasReferences()
                 << '\n';
        }
      }

      return CalleeTracker;
    }

    bool hasRefAt(const DSNode *N, unsigned Offset) const {
      if (N->hasLink(Offset) && Targets.count(N->getLink(Offset).getNode()))
        return true;

      auto It = RefRecords.find(N);
      if (It != RefRecords.end())
        return It->second.hasRefAt(Offset);

      return false;
    }

    bool refersTargets(const DSNode *N) const {
      for (auto ei = N->edge_begin(), ee = N->edge_end(); ei != ee; ++ei)
        if (Targets.count(ei->second.getNode()))
          return true;

      auto RefIt = RefRecords.find(N);
      if (RefIt != RefRecords.end())
        return RefIt->second.hasReferences();

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

  Type *VoidTy, *Int8Ty, *Int32Ty, *Int64Ty;
  StructType *TripleTy;
  PointerType *Int8PtrTy, *TriplePtrTy;
  Constant *Zero, *One, *Two, *Zero64, *One64;

  std::set<const DSNode *> findTransformableNodes(const Module &M);

  // mapping value -> value it replaced
  std::map<const Value *, const Value *> ReplacedValueMap;

  const Value *getOrigValue(const Value *V) const {
    auto ValIt = ReplacedValueMap.find(V);
    if (ValIt != ReplacedValueMap.end())
      return ValIt->second;
    return V;
  }

  void updateValue(const Value *Old, const Value *NewV) {
    ReplacedValueMap[NewV] = getOrigValue(Old);
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
      Module &M, 
      const TargetMapTy &TransTargets, const TargetMapTy &TargetsInGG,
      std::map<unsigned, OffsetMapTy> &OffsetMaps,
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

  FunctionType *computeCloneFnTy(CallSite CS,
                                 const RefSetTracker<TargetMapTy> &);
  Function *cloneFunctionWithNewType(Function *Src, FunctionType *FnTy,
                                     const TargetMapTy &Targets,
                                     std::map<Value *, IndexTriple> &TripleMap);
  std::vector<DSGraph *> SortedSCCs;
  std::map<DSGraph *, unsigned> SCCToOrderMap;
  std::map<DSGraph *, std::vector<CallEdgeTy>> OutgoingEdges;
  void buildMetaGraph(Module &M);
  bool safeToTransform(const NodeSetTy &TransTargets, OffsetMapTy &OffsetMap,
                       std::map<DSGraph *, NodeSetTy> &TargetMap,
                       const Module &M);

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
  StructType *computeTypeWithTripleConv(DSNodeHandle NH,
                                        const NODE_SET_TY &Targets) const;

  template <typename NODE_SET_TY>
  unsigned computeSizeWithTripleConv(DSNodeHandle NH,
                                     const NODE_SET_TY &Targets) const;

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
  if (!isa<PointerType>(V->getType())) return false;

  // Undef values, even ones of pointer type, don't get nodes.
  if (isa<UndefValue>(V)) return false;

  if (isa<ConstantPointerNull>(V))
    return false;

  // Use the Aliasee of GlobalAliases
  // FIXME: This check might not be required, it's here because
  // something similar is done in the Local pass.
  if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(V))
    return shouldHaveNodeForValue(GA->getAliasee());

  return true;
}

} // end anonymous namespace

char LayoutTuner::ID = 42;

// define 'initializeLayoutTunerPass()'
INITIALIZE_PASS_BEGIN(LayoutTuner, "", "", true, true)
initializeTypeSafetyEquivBU(Registry);
INITIALIZE_PASS_DEPENDENCY(EquivBUDataStructures)
INITIALIZE_PASS_DEPENDENCY(AllocIdentify)
INITIALIZE_PASS_END(LayoutTuner, "", "", true, true)

TransformLowering::TransformLowering(const DSNode *Target) {
  std::vector<Type *> Fields;
  unsigned i = 0;
  for (auto ti = Target->type_begin(), te = Target->type_end(); ti != te;
       ++ti, ++i) {
    Type *FieldTy = *ti->second->begin();
    Fields.push_back(FieldTy);
    Offset2IdxMap[ti->first] = i;
  }
  LLVMContext &Ctx = Fields[0]->getContext();
  TargetTy = StructType::get(Ctx, Fields);
  ElemSize = ConstantInt::get(Type::getInt64Ty(Ctx), Target->getSize());
  errs() << "TARGET TYPE: " << *TargetTy << '\n';
}

Value *NoopTransform::ComputeAddress(IndexTriple Triple, unsigned Offset,
                                     Instruction *InsertPt) {
  auto *Int32Ty = Type::getInt32Ty(InsertPt->getContext());
  assert(Offset2IdxMap.count(Offset) && "invalid offset used for indexing");
  auto *FieldIdx = ConstantInt::get(Int32Ty, Offset2IdxMap.at(Offset));
  auto *Base = CastInst::CreatePointerCast(
      Triple.Base, PointerType::getUnqual(TargetTy), "", InsertPt);
  return GetElementPtrInst::Create(
      TargetTy, Base, GEPIdxTy{Triple.Idx, FieldIdx}, "", InsertPt);
}

LayoutLowering::LayoutLowering(const DSNode *Target,
                               std::shared_ptr<LayoutDataType> TheLayout,
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

  Visit(-1, TheLayout.get());

  assert(Paths.size() == Offset2IdxMap.size());
}

Value *LayoutLowering::ComputeAddress(IndexTriple Triple, unsigned Offset,
                                      Instruction *InsertPt) {
  assert(Offset2IdxMap.count(Offset) && "unknown offset");
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

  GEPIdxTy GEPIdxs { CurNode->getIdxExpr()->emitCode(Variables, InsertPt) };
  for (; PathIdx < PathLen; PathIdx++) {
    auto *Node = Path[PathIdx];
    auto Dims = Node->getDims();

    // find idx expression for array indexing
    // Skip this for CurNode for which we've emitted a linearized index expression
    if (Node != CurNode) {
      for (auto DI = Dims.rbegin(), DE = Dims.rend(); DI != DE; ++DI) 
        GEPIdxs.push_back(DI->second->emitCode(Variables, InsertPt));
    }

    // find idx for struct offset
    if (auto *Struct = dyn_cast<LayoutStruct>(Node)) {
      int Offset = -1, i = 0;
      auto Fields = Struct->getFields();
      auto *NextNode = Path[PathIdx+1];
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
  errs() << "GEP IDXS:\n";
  for (auto *Idx : GEPIdxs)
    errs() << '\t'<< *Idx << '\n';
  errs() << *ETy << '\n';
  auto *Ptr = CastInst::CreatePointerCast(Addr, PointerType::getUnqual(ETy), "", InsertPt);
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
StructType *
LayoutTuner::computeTypeWithTripleConv(DSNodeHandle NH,
                                       const NODE_SET_TY &Targets) const {
  auto *N = NH.getNode();
  std::vector<Type *> Elems;
  for (auto ti = N->type_begin(), te = N->type_end(); ti != te; ++ti) {
    unsigned Offset = ti->first + NH.getOffset();
    Type *ET = *ti->second->begin();
    if (N->hasLink(Offset) && Targets.count(N->getLink(Offset).getNode()))
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
unsigned
LayoutTuner::computeSizeWithTripleConv(DSNodeHandle NH,
                                       const NODE_SET_TY &Targets) const {

  auto *ConvertedType = computeTypeWithTripleConv(NH, Targets);
  return TD->getTypeAllocSize(ConvertedType);
}

//
// clone a function and change it to use arguments properly
//
Function *LayoutTuner::cloneFunctionWithNewType(
    Function *Src, FunctionType *FnTy, const TargetMapTy &Targets,
    std::map<Value *, IndexTriple> &TripleMap) {
  ValueToValueMapTy ValueMap;
  auto *Clone = CloneFunction(Src, ValueMap);
  Clone->setVisibility(Function::DefaultVisibility);
  Clone->setLinkage(Function::InternalLinkage);

  auto *NewF = Function::Create(FnTy, Function::InternalLinkage);

  auto &CloneRecord = CloneRecords[NewF];
  CloneCache[{Targets, getSrcFunc(Src)}] = NewF;

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
  NewF->takeName(Clone);
  NewF->getBasicBlockList().splice(NewF->begin(), Clone->getBasicBlockList());

  // transfer arguments from Clone to NewF
  for (auto I = Clone->arg_begin(), E = Clone->arg_end(),
            I2 = NewF->arg_begin();
       I != E; ++I, ++I2) {
    auto *N = getNodeForValue(&*I, NewF).getNode();
    if (Targets.count(N)) {
      // pointer to target has been expanded into triple
      auto *Base = &*I2++;
      auto *Idx = &*I2++;
      auto *Size = &*I2;
      errs() << "Setting triple for " << &*I << "\n";
      Base->setName("base");
      Idx->setName("idx");
      Size->setName("size");
      TripleMap[&*I] = {Base, Idx, Size};
    } else {
      CloneRecord.CloneToSrcValueMap[&*I2] =
          CloneRecord.CloneToSrcValueMap[&*I];
      I->mutateType(I2->getType());
      I->replaceAllUsesWith(&*I2);
      I2->takeName(&*I);
    }
  }

  removeValue(Clone);
  return NewF;
}

bool LayoutTuner::isAllocator(const Function *F) {
  // return std::any_of(
  //    AllocIdentifier->alloc_begin(), AllocIdentifier->alloc_end(),
  //    [&](const std::string Name) {
  //    return F->getName() == Name; });
  StringRef Name = F->getName();
  return Name == "malloc" || Name == "calloc";
}

bool LayoutTuner::isDeallocator(const Function *F) {
  // return std::any_of(
  //    AllocIdentifier->dealloc_begin(), AllocIdentifier->dealloc_end(),
  //    [&](const std::string Name) { return F->getName() == Name; });
  StringRef Name = F->getName();
  return Name == "free";
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

  TargetMapTy Targets;
  auto Candidates = findTransformableNodes(M);

  std::map<unsigned, OffsetMapTy> OffsetMaps;
  std::map<const DSNode *, std::map<DSGraph *, NodeSetTy>> TargetMaps;

  errs() << "MIN SIZE IS " << MinSize << '\n';

  unsigned GroupIdx = 0;
  for (auto *Target : Candidates) {
    auto &OffsetMap = OffsetMaps[GroupIdx];
    auto &TargetMap = TargetMaps[Target];
    errs() << "=============\n";
    if (Target->getSize() > MinSize &&
        safeToTransform({Target}, OffsetMap, TargetMap, M)) {
      Targets[Target] = GroupIdx;
      errs() << "Maybe " << GroupIdx << " is good\n";
      errs() << "Size of offset map " << OffsetMap.size() << '\n';
      errs() << "Size of target map " << TargetMap.size() << '\n';
    }

    ++GroupIdx;
  }

  //
  // Remove nodes that are covered by other nodes from a predecessor
  // FIXME: this is not correct, considering functions in the same SCC might not
  // share the same DSGraph
  //
  std::set<const DSNode *> Targets2Remove;
  for (auto TargetAndGroup : Targets) {
    auto *Target = TargetAndGroup.first;
    if (Targets2Remove.count(Target))
      continue;

    auto &TargetMap = TargetMaps[Target];
    unsigned GraphIdx = SCCToOrderMap[Target->getParentGraph()];
    assert(SortedSCCs[GraphIdx] == Target->getParentGraph());
    for (unsigned i = GraphIdx + 1, e = SortedSCCs.size(); i != e; i++) {
      auto *G = SortedSCCs[i];
      for (auto *N : TargetMap[G]) {
        if (Targets.count(N))
          errs() << "DAFUQ " << Targets.at(N) << " is covered by " << Targets.at(Target) << '\n';
      }
      Targets2Remove.insert(TargetMap[G].begin(), TargetMap[G].end());
    }
  }

  for (auto *Node : Targets2Remove) {
    Targets.erase(Node);
  }

  //
  // the transformation doesn't work (yet?) when one target refers another or
  // itself
  // hold on, maybe it works!
  // FIXME: let me get back to this later
  //
  bool Changed;
  do {
    Changed = false;
    for (auto TargetAndGroup : Targets) {
      auto *Target = TargetAndGroup.first;
      auto *G = Target->getParentGraph();
      for (auto TargetAndGroup2 : Targets) {
        auto *Target2 = TargetAndGroup2.first;
        if (refersTargets(Target, TargetMaps[Target2][G])) {
          Changed = true;
          Targets.erase(Target);
          break;
        }
      }
      if (Changed)
        break;
    }
  } while (Changed);

  // mappping trasform group -> the group's layout
  std::map<unsigned, std::shared_ptr<TransformLowering>> Transforms;

  TargetMapTy TargetsInGG;

  TransformPool TP;
  TP.addTransform(std::make_unique<FactorTransform>(4), 0.05);
  TP.addTransform(std::make_unique<FactorTransform>(8), 0.05);
  TP.addTransform(std::make_unique<FactorTransform>(16), 0.05);
  TP.addTransform(std::make_unique<FactorTransform>(32), 0.05); // 0.2
  TP.addTransform(std::make_unique<StructTransform>(), 0.15); // 0.35
  TP.addTransform(std::make_unique<StructFlattenTransform>(), 0.15); // 0.5
  TP.addTransform(std::make_unique<SOATransform>(), 0.15); // 0.65
  TP.addTransform(std::make_unique<AOSTransform>(), 0.15); // 0.8
  TP.addTransform(std::make_unique<SwapTransform>(), 0.05); // 0.85
  TP.addTransform(std::make_unique<InterchangeTransform>(), 0.05); // 0.85

  auto transform = [&](LayoutDataType *Layout) -> std::unique_ptr<LayoutDataType> {
    std::vector<std::unique_ptr<LayoutDataType>> Tmps;
    Tmps.push_back(TP.apply(*Layout));
    for (int i = 0; i < 7; i++)
      Tmps.push_back(TP.apply(*Tmps.back()));
    return std::move(Tmps.back());
  };

  unsigned NumUniqueNodes = 0;
  for (auto TargetAndGroup : Targets) {
    const DSNode *Target;
    unsigned Group;
    std::tie(Target, Group) = TargetAndGroup;

    const auto &GlobalTargets =
        TargetMaps.at(Target).at(DSA->getGlobalsGraph());
    assert(GlobalTargets.size() <= 1 &&
           "one-to-many mapping from local graph to global graph?");

    if (!GlobalTargets.empty()) {
      auto *N = *GlobalTargets.begin();
      if (TargetsInGG.count(N)) {
        // this target is actually the same object as one of the targets we
        // encountered before
        // so, they *must* use the same transformation
        // TODO: merge transform group properly
        Transforms[Group] = Transforms.at(TargetsInGG.at(N));
        errs() << "MERGED GROUP " << Group << '\n';
      } else {
        TargetsInGG[N] = Group;
        // Transforms[Group] = std::make_shared<NoopTransform>(Target);
        auto SizeVar = getNewName(), IdxVar = getNewName();
        auto OrigLayout = std::shared_ptr<LayoutStruct>(
            LayoutStruct::create(Target, TD, SizeVar, IdxVar));
        Transforms[Group] = std::make_shared<LayoutLowering>(
            Target, transform(OrigLayout.get()), SizeVar, IdxVar,
            TD);
        errs() << "ADDR OF TRANSFORM: " << Transforms[Group].get() << '\n';
        NumUniqueNodes++;
      }
    } else {
      // Transforms[Group] = std::make_shared<NoopTransform>(Target);
      auto SizeVar = getNewName(), IdxVar = getNewName();
      auto OrigLayout = std::shared_ptr<LayoutStruct>(
          LayoutStruct::create(Target, TD, SizeVar, IdxVar));
      Transforms[Group] = std::make_shared<LayoutLowering>(
          Target, transform(OrigLayout.get()), SizeVar, IdxVar,
          TD);
      errs() << "ADDR OF TRANSFORM: " << Transforms[Group].get() << '\n';
      NumUniqueNodes++;
    }
  }

  errs() << "Number of unique nodes safe to transform: " << NumUniqueNodes
         << '\n';

  applyTransformation(M, Targets, TargetsInGG, OffsetMaps, Transforms);

  CalleeCallerMappings.clear();
  GToGGMappings.clear();

  cleanupDeadValues();
  M.dump();

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
  if (!N || N->isCollapsedNode() || Offset >= N->getSize())
    return CurTy;

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
      if (Refs.hasRefAt(N, ElementOffset))
        NewET = TripleTy;
      else if (isa<PointerType>(ET) && N->hasLink(ElementOffset)) {
        // pointer!
        auto &Cell = N->getLink(ElementOffset);
        NewET = getNewType(Cell.getNode(), Refs, ET, Cell.getOffset());
      } else
        NewET = getNewType(N, Refs, ET, Offset);

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

  return getNewType(N, Refs, V->getType(), Offset);
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
  errs() << "---- loading triple.base\n";
  // load base ptr
  auto *BaseAddr = GetElementPtrInst::Create(
      TripleTy, Ptr, GEPIdxTy{Zero, Zero}, "base_addr", InsertPt);
  auto *Base = new LoadInst(BaseAddr, "base", InsertPt);

  errs() << "---- loading triple.idx\n";
  // load idx
  auto *IdxAddr = GetElementPtrInst::Create(TripleTy, Ptr, GEPIdxTy{Zero, One},
                                            "idx_addr", InsertPt);
  auto *Idx = new LoadInst(IdxAddr, "Idx", InsertPt);

  errs() << "---- loading triple.size\n";
  // load offset
  auto *SizeAddr = GetElementPtrInst::Create(TripleTy, Ptr, GEPIdxTy{Zero, Two},
                                             "size_addr", InsertPt);
  auto *Size = new LoadInst(SizeAddr, "size", InsertPt);

  return {Base, Idx, Size};
}

void LayoutTuner::storeTriple(IndexTriple T, Value *Ptr,
                              Instruction *InsertPt) const {
  errs() << "--- Storing triple.base\n";
  auto *BaseAddr = GetElementPtrInst::Create(
      TripleTy, Ptr, GEPIdxTy{Zero, Zero}, "base_addr", InsertPt);
  new StoreInst(T.Base, BaseAddr, InsertPt);

  errs() << "--- Storing triple.idx\n";
  auto *IdxAddr = GetElementPtrInst::Create(TripleTy, Ptr, GEPIdxTy{Zero, One},
                                            "idx_addr", InsertPt);
  new StoreInst(T.Idx, IdxAddr, InsertPt);

  errs() << "--- Storing triple.size\n";
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

  //FIXME: Here we trust the signature of the callsite to determine which arguments
  //are var-arg and which are fixed.  Apparently we can't assume this, but I'm not sure
  //of a better way.  For now, this assumption is known limitation.
  const FunctionType *CalleeFuncType = DSCallSite::FunctionTypeOfCallSite(CS);
  int NumFixedArgs = CalleeFuncType->getNumParams();
  
  // Sanity check--this really, really shouldn't happen
  if (!CalleeFuncType->isVarArg())
    assert(CS.arg_size() == static_cast<unsigned>(NumFixedArgs) &&
        "Too many arguments/incorrect function signature!");

  std::vector<DSNodeHandle> Args;
  Args.reserve(CS.arg_end()-CS.arg_begin());

  // Calculate the arguments vector...
  for (CallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end(); I != E; ++I)
    if (isa<PointerType>((*I)->getType())) {
      DSNodeHandle ArgNode; // Initially empty
      if (shouldHaveNodeForValue(*I)) ArgNode = getNodeForValue(I->get(), F);
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
  if (Function *Callee=dyn_cast<Function>(CS.getCalledValue()->stripPointerCasts()))
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
  DSCallSite DSCS = getDSCallSiteForCallSite(CallSite(const_cast<Instruction *>(I)));
  DSGraph *CalleeG = DSA->getDSGraph(*Callee);

  errs() << "COMPUTING CALLEE CALLER MAPPING\n";
  G->computeCalleeCallerMapping(DSCS, *Callee, *CalleeG, CalleeCallerMapping,
                                false);

  return CalleeCallerMapping;
}

//
// i am too lazy to support these now
//
static bool isUnsupportedLibCall(const Function *F) {
  auto IID = F->getIntrinsicID();
  return IID == Intrinsic::memcpy || IID == Intrinsic::memmove ||
         IID == Intrinsic::memset || F->getName() == "realloc";
}

//
// is the set of DSNodes `TransTargets` safe to transform?
//
// FIXME:
// this is broken. We also need to check a
bool LayoutTuner::safeToTransform(const NodeSetTy &TransTargets,
                                  OffsetMapTy &OffsetMap,
                                  std::map<DSGraph *, NodeSetTy> &TargetMap,
                                  const Module &M) {
  assert(TargetMap.empty());
  if (TransTargets.empty())
    return true;

  const DSNode *SomeTarget = *TransTargets.begin();
  unsigned TargetSize = SomeTarget->getSize();

  for (auto *N : TransTargets) {
    TargetMap[N->getParentGraph()].insert(N);
    OffsetMap[N].insert(0);
  }

  auto &TypeChecker = getAnalysis<TypeSafety<EquivBUDataStructures>>();

  auto *GG = DSA->getGlobalsGraph();

  // propagate targets and offsets
  // treat this a dataflow problem
  bool Changed;
  do {
    Changed = false;

    for (auto *G : SortedSCCs) {
      const NodeSetTy &CurTargets = TargetMap[G];

      // nothing to check or propagate
      if (CurTargets.empty())
        continue;

      //
      // TODO: handle cases where offset is is not static by fusing contiguous
      // fields as single scalar
      //
      // For now, if there's ambiguity, just give up
      //
      for (auto *Target : CurTargets)
        if (OffsetMap[Target].size() != 1) {
          errs() << "Can't transform target: multiple offsets\n";
          return false;
        }

      //
      // make sure no type-unsafe node is pointing to the target
      //
      for (auto ni = G->node_begin(), ne = G->node_end(); ni != ne; ++ni) {
        const DSNode *N = &*ni;
        if (!TypeChecker.isTypeSafeIfInternal(N) &&
            refersTargets(N, CurTargets)) {
          errs() << "Can't transform target: referred by unsafe node\n";
          return false;
        }
      }

      //
      // propagate offsets and targets
      //
      for (auto Edge : OutgoingEdges[G]) {
        const auto &CalleeCallerMapping = getCalleeCallerMapping(Edge);
        auto *CalleeG = DSA->getDSGraph(*Edge.second);
        errs() << "PROCESSING EDGE " << Edge.first->getParent()->getParent()->getName() << " -> " << Edge.second->getName() << '\n';
        for (auto Mapping : CalleeCallerMapping) {
          auto *CalleeN = Mapping.first;
          auto *CallerN = Mapping.second.getNode();
          if (!CurTargets.count(CallerN))
            continue;

          errs() << "PROP " << CallerN << " -> " << CalleeN << '\n';

          unsigned RelOffset = Mapping.second.getOffset();
          auto &PotentialOffsets = OffsetMap[CalleeN];
          assert(OffsetMap.count(CallerN));
          assert(!OffsetMap.at(CallerN).empty());
          for (unsigned Offset : OffsetMap[CallerN]) {
            unsigned AbsOffset = Offset + RelOffset;
            if (AbsOffset >= TargetSize)
              return false;
            Changed |= PotentialOffsets.insert(AbsOffset).second;
          }

          Changed |= TargetMap[CalleeG].insert(CalleeN).second;
        }
      }

      //
      // propagate targets to globals graph
      //
      const auto &NodeMap = getGToGGMapping(G);
      for (auto *Target : CurTargets) {
        auto Mapping = NodeMap.find(Target);
        if (Mapping != NodeMap.end()) {
          auto GGNH = Mapping->second;
          Changed |= TargetMap[GG].insert(GGNH.getNode()).second;
          assert(OffsetMap.count(Target));
          assert(!OffsetMap.at(Target).empty());
          OffsetMap[GGNH.getNode()].insert(
              *OffsetMap.at(Target).begin() + GGNH.getOffset());
          if (OffsetMap.at(GGNH.getNode()).size() != 1) {
            errs() << "Can't transform target: multiple offsets\n";
            return false;
          }
        }
      }
    }
  } while (Changed);

  auto &Targets = TargetMap[GG];
  for (auto &G : M.globals()) {
    auto NH = GG->getNodeForValue(&G);
    auto *N = NH.getNode();
    if (N && refersTargets(N, Targets)) {
      // for now, to simplify transformation,
      // we support handling global that refers to target
      // only if the global is declared with type infered by DSA
      unsigned SizeNeeded = computeSizeWithTripleConv(N, Targets);
      auto *TyAfterConv =
          cast<PointerType>(getNewType(const_cast<GlobalVariable *>(&G), NH,
                                       RefSetTracker<NodeSetTy>(Targets)))
              ->getElementType();
      if (TD->getTypeAllocSize(TyAfterConv) != SizeNeeded)
        return false;

      // for now only allow such globals to be used in instructions directly
      for (auto *U : G.users())
        if (isa<Constant>(U))
          return false;
    }
  }
  errs() << "I am here\n";

  auto &CG = DSA->getCallGraph();

  // make sure that, if an element, B, if derived from another element, A, in an
  // array, it can *only* be derived with
  // `getelementptr <type with size of target>, A, ...` A's offset is 0.
  //
  // also make sure no indirect call could access a target
  for (auto &F : M) {
    if (F.empty())
      continue;

    auto *G = getDSGraph(&F);
    const auto &Targets = TargetMap[G];
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          auto SrcNH = getNodeForValue(GEP->getPointerOperand(), &F);
          if (!Targets.count(SrcNH.getNode()))
            continue;

          auto *SrcN = SrcNH.getNode();
          unsigned SrcSize = TD->getTypeAllocSize(GEP->getSourceElementType());
          auto &AbsOffsets = OffsetMap.at(SrcN);
          bool CanChangeIdx =
              (SrcSize == TargetSize && AbsOffsets.size() == 1 &&
               (*AbsOffsets.begin()) == 0);

          auto *Idx = GEP->idx_begin()->get();
          auto *ConstIdx = dyn_cast<ConstantInt>(Idx);
          if (!CanChangeIdx) {
            if (!ConstIdx)
              return false;

            APInt RelOffset(TD->getPointerSizeInBits(GEP->getAddressSpace()),
                            0);
            bool HasConstOffset = GEP->accumulateConstantOffset(*TD, RelOffset);
            if (!HasConstOffset)
              return false;
            for (unsigned AbsOffset : AbsOffsets)
              if ((AbsOffset + RelOffset).getSExtValue() >= TargetSize)
                return false;
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

            for (auto *N : ReachableNodes)
              if (Targets.count(N) || refersTargets(N, Targets))
                return false;
          }

          //
          // make sure we don't pass a target to memcpy, memmove, and memset
          //
          bool TargetUsedInCall =
              Targets.count(G->getNodeForValue(&I).getNode()) > 0;
          for (auto &Arg : CS.args()) {
            auto *N = G->getNodeForValue(Arg.get()).getNode();
            if (Targets.count(N)) {
              TargetUsedInCall = true;
              break;
            }
          }

          if (!TargetUsedInCall)
            continue;

          auto *Callee = CS.getCalledFunction();
          bool HasDefinition = isDeallocator(Callee) || isAllocator(Callee) || !Callee->isDeclarationForLinker();
          if (Callee &&
              (!HasDefinition || isUnsupportedLibCall(Callee))) {
            return false;
          }
        }
  }

  return true;
}

// TODO: finish this and get a drink
// TODO: handle memcpy, memmove, and memset
// TODO: deal with recursive data structure
// TODO: handle cases where one targets points to another...
// delay code-gen for address calculation)
// TODO: handle indirect call
//
// TODO: try *really hard* to think about why the way we processing function works (I think?)
// -- push orig functions on to worklist first, and don't process functions that should be clone
//
// FIXME: Should record size of target/target-ref on top level, since once once
// one descend down
// the call graph, DSNode::getSize is not accurate
void LayoutTuner::applyTransformation(
    Module &M, const TargetMapTy &TransTargets,
    const TargetMapTy &TargetsInGG,
    std::map<unsigned, LayoutTuner::OffsetMapTy> &OffsetMaps,
    std::map<unsigned, std::shared_ptr<TransformLowering>> &Transforms) {

  // mapping global that points to targets -> same globals re-declared with proper types
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

  std::vector<std::pair<Function *, RefSetTracker<TargetMapTy>>> Worklist;

  // scan the call graph top-down and rewrite functions along the way
  for (auto gi = SortedSCCs.rbegin(), ge = SortedSCCs.rend();
      gi != ge; ++gi) {
    DSGraph *G = *gi;
    for (auto ri = G->retnodes_begin(), re = G->retnodes_end();
        ri != re; ++ri) {
      auto *F = const_cast<Function *>(ri->first);
      Worklist.push_back({ F, TransTargets });
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

  while (!Worklist.empty()) {
    Function *F = Worklist.back().first;
    auto Refs = std::move(Worklist.back().second);
    auto &Targets = Refs.getTargets();
    Worklist.pop_back();

    errs() << "!! processing " << F->getName() << '\n';
    errs() << "!!! there are " << Targets.size() << " targets\n";

    if (F->empty())
      continue;

    if (Processed[F])
      continue;
    Processed[F] = true;

    // FIXME: why is this necesary
    //
    // propagate targets from global graph
    auto &GToGGMapping = getGToGGMapping(getDSGraph(F));
    for (auto &Mapping : GToGGMapping) {
      auto It = TargetsInGG.find(Mapping.second.getNode());
      if (It != TargetsInGG.end()) {
        unsigned GroupId = It->second;
        Targets[Mapping.first] = GroupId;
        unsigned Offset = *OffsetMaps.at(GroupId).at(Mapping.second.getNode()).begin() +
          Mapping.second.getOffset();
        OffsetMaps[GroupId][Mapping.first].insert(Offset);
      }
    }

    // F will be dead after cloning anyways
    if (!CloneRecords.count(F) &&
        shouldCloneFunction(F, Refs))
        continue;

    //
    // Mapping a value to a set of <phi, basic block>, where it's used
    // we remember values whose triples we need when we emit phis but are
    // not computed at the time. We go back to fix them up later.
    //
    std::map<Value *, std::vector<IncomingTripleTy>> IncomingValues;

    for (auto &BB : *F) {
      auto I = BB.begin(), E = BB.end(), NextIt = std::next(I);
      for (; I != E; I = NextIt, ++NextIt) {
        auto NH = getNodeForValue(&*I, F);
        auto *N = NH.getNode();

        assert(!isa<ExtractValueInst>(&*I) && !isa<InsertValueInst>(&*I) &&
               "doesn't support extractvalue or insertvalue yet");

        for (unsigned i = 0, e = I->getNumOperands();
            i != e; i++) {
          auto &Op = I->getOperandUse(i);
          auto Redeclared = RedeclaredGlobals.find(Op.get());
          if (Redeclared != RedeclaredGlobals.end())
            Op.set(Redeclared->second);
        }

        auto *GEP = dyn_cast<GetElementPtrInst>(&*I);
        if (GEP && N) {
          errs() << "--- handling gep\n";
          Value *Src = GEP->getPointerOperand();
          Type *OrigSrcTy = GEP->getSourceElementType();
          Type *SrcTy = getNewType(Src, getNodeForValue(Src, F), Refs);
          Type *ResultTy = getNewType(GEP, getNodeForValue(GEP, F), Refs);
          errs() << "--- Deduced type for gep\n";

          if (Targets.count(N)) {
            errs() << "--- updating idx\n";
            // Src is Target
            // we postpone the actual address calculation by updating the triple
            assert(TripleMap.count(Src));
            IndexTriple &OldTriple = TripleMap.at(Src);
            IndexTriple &NewTriple = TripleMap[GEP] = OldTriple;
            Instruction *InsertPt = &*std::next(I);

            // Update Idx
            if (TD->getTypeAllocSize(OrigSrcTy) == N->getSize()) {
              auto *Diff = GEP->idx_begin()->get();
              if (Diff->getType() != Int64Ty)
                Diff = new ZExtInst(Diff, Int64Ty, "", InsertPt);
              NewTriple.Idx = BinaryOperator::CreateNSWAdd(
                  OldTriple.Idx, Diff, "updated_idx", InsertPt);
            }

            removeValue(GEP);
          } else {
            //
            // rewrite the gep with correct types
            //
            GEP->setSourceElementType(
                cast<PointerType>(SrcTy)->getElementType());
            GEP->setResultElementType(
                cast<PointerType>(ResultTy)->getElementType());
            auto *SrcFixed = CastInst::CreatePointerCast(Src, SrcTy, "", GEP);
            updateValue(Src, SrcFixed);
            GEP->getOperandUse(0).set(SrcFixed);
            GEP->mutateType(ResultTy);
          }
        } // end of handling GEP

        if (auto *LI = dyn_cast<LoadInst>(&*I)) {
          errs() << "--- handling load\n";
          if (Targets.count(N)) {
            errs() << "--- loading triple-converted pointer\n";
            // case 1: loading pointer to Target
            // instead of loading the pointer, we load the triple
            Instruction *InsertPt = &*std::next(I);
            errs() << "--- Setting triple for " << LI << '\n';
            auto *TriplePtr = CastInst::CreatePointerCast(
                LI->getPointerOperand(), TriplePtrTy, "", InsertPt);
            TripleMap[LI] = loadTriple(TriplePtr, InsertPt);
            removeValue(LI);
          }

          auto PointerNH = getNodeForValue(LI->getPointerOperand(), F);
          if (Targets.count(PointerNH.getNode())) {
            errs() << "--- loading from triple-converted pointer\n";
            // case 2: loading from Target itself
            // we need to replace the load, since layout of the allocation has
            // changed
            //
            // assuming the data structure is not recursive,
            // we forward address calculation of to whatever transformation we
            // are trying to apply,
            // which knows the precise layout
            Instruction *InsertPt = &*std::next(I);

            unsigned GroupId = Targets.at(PointerNH.getNode());
            unsigned Offset =
                *OffsetMaps.at(GroupId).at(PointerNH.getNode()).begin() +
                PointerNH.getOffset();
            errs() << "-- replacing load\n";
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
          errs() << "--- handling store\n";
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
            errs() << *Src << '\n';
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
            unsigned GroupId = Targets.at(DestN);
            unsigned Offset =
                *OffsetMaps.at(GroupId).at(DestN).begin() + DestNH.getOffset();
            auto *Addr = Transforms.at(GroupId)->ComputeAddress(
                TripleMap.at(Dest), Offset, InsertPt);
            auto *Ptr = CastInst::CreatePointerCast(Addr, Dest->getType(),
                                                    I->getName(), InsertPt);
            updateValue(Dest, Addr);
            updateValue(Dest, Ptr);
            new StoreInst(Src, Ptr, InsertPt);
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
              PN->mutateType(NewType);
              for (unsigned i = 0; i < PN->getNumIncomingValues(); i++) {
                auto *Incoming = PN->getIncomingValue(i);
                if (isa<UndefValue>(Incoming))
                  Incoming->mutateType(NewType);
                else if (Incoming->getType() != NewType) {
                  auto *IncomingBB = PN->getIncomingBlock(i);
                  auto *IncomingFixed = CastInst::CreatePointerCast(
                      Incoming, NewType, "", IncomingBB->getTerminator());
                  PN->setIncomingValue(i, IncomingFixed);
                }
              }
            }
            continue;
          }
          errs() << "~~~~~ " << N << '\n';
          errs() << "--- handling phi\n";
          unsigned NumIncomings = PN->getNumIncomingValues();
          Instruction *InsertPt = &*I;
          auto *BasePHI =
              PHINode::Create(Int8PtrTy, NumIncomings, "base", InsertPt);
          auto *IdxPHI =
              PHINode::Create(Int64Ty, NumIncomings, "idx", InsertPt);
          auto *SizePHI =
              PHINode::Create(Int64Ty, NumIncomings, "size", InsertPt);

          errs() << "Setting triple for " << PN << '\n';
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
          errs() << "--- handling cast\n";
          auto *Src = Cast->getOperand(0);
          errs() << "Copying triple from " << Src << " to " << Cast << '\n';
          if (!TripleMap.count(Src)) {
            errs() << "!! " << *Src << '\n';
          }
          assert(TripleMap.count(Src));
          assert(getNodeForValue(Src, F).getNode() == N);
          TripleMap[Cast] = TripleMap.at(Src);
          removeValue(Cast);
        } // end of handling bitcast

        if (auto CS = CallSite(&*I)) {
          // ignore indirect call for now
          if (!CS.getCalledFunction())
            continue;

          Function *Callee = CS.getCalledFunction();
          // rewrite the allocation
          if (isAllocator(Callee)) {
            errs() << "ALLOCATOR: " << Callee->getName() << '\n';
            assert(CS.getNumArgOperands() == 1 ||
                   Callee->getName() == "calloc" && "Unknown allocator");
            auto *TotalSize = CS.getArgument(0);
            // OMFG
            if (Callee->getName() == "calloc")
              TotalSize = BinaryOperator::CreateNSWMul(
                  TotalSize, CS.getArgument(1), "", &*I);

            if (Targets.count(N)) {
              errs() << "--- rewriting alloc\n";
              //
              // rewrite the allocation of type whose layout we changed
              //
              unsigned GroupId = Targets.at(N);
              Value *Size, *SizeInBytes, *Base;
              std::tie(Size, SizeInBytes) = computeAllocSize(
                  TotalSize, N, Transforms.at(GroupId).get(), &*I);
              Base = rewriteAlloc(CS, SizeInBytes, &*I, "base");

              // fingers crossed an integer is either 64 bit or less
              if (Size->getType() != Int64Ty)
                Size = new ZExtInst(Size, Int64Ty, "size", &*I);

              errs() << "Setting triple for " << &*I << '\n';
              TripleMap[&*I] = {Base, Zero64, Size};
              removeValue(&*I);
            } else if (N && !N->isCollapsedNode() && Refs.refersTargets(N)) {
              //
              // rewrite the allocation of type that got triple-converted
              //
              unsigned SizeNeeded = computeSizeWithTripleConv(NH, Targets);
              if (SizeNeeded != N->getSize()) {
                assert(SizeNeeded > N->getSize());
                auto *SizeTy = TotalSize->getType();
                Value *NumElems = BinaryOperator::CreateExactUDiv(
                    TotalSize, ConstantInt::get(SizeTy, N->getSize()),
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
              errs() << "--- rewriting dealloc\n";
              assert(CS.getNumArgOperands() == 1);
              std::vector<Value *> DeallocArgs = {Ptr};
              if (auto *Invoke = dyn_cast<InvokeInst>(CS.getInstruction()))
                InvokeInst::Create(Callee, Invoke->getNormalDest(),
                                   Invoke->getUnwindDest(), DeallocArgs, "",
                                   &*I);
            }
            removeValue(&*I);
            continue;
          } // end of handling allocation

          if (Callee->empty())
            continue;

          errs() << "--- handling function call to " << Callee->getName()
                 << '\n';

          auto *CallerG = getDSGraph(F), *CalleeG = getDSGraph(Callee);
          // propagate targets from caller to callee
          // also propagate the group tag
          TargetMapTy TargetsInCallee = TransTargets;
          RefSetTracker<TargetMapTy> CalleeRefs;
          errs() << "Caller Graph: " << CallerG << ", Callee Graph" << CalleeG
                 << '\n';
          if (CallerG != CalleeG) {
            bool SameSCC = (DSA->getCallGraph().sccLeader(getSrcFunc(F)) ==
                            DSA->getCallGraph().sccLeader(getSrcFunc(Callee)));
            errs() << F->getName() << " and " << Callee->getName()
                   << " in the same SCC: " << SameSCC << '\n';
            const auto &CalleeCallerMapping =
                getCalleeCallerMapping({CS, Callee});
            for (auto Mapping : CalleeCallerMapping) {
              auto *CalleeN = Mapping.first;
              auto *CallerN = Mapping.second.getNode();

              assert(!SameSCC || Mapping.second.getOffset() == 0);

              auto It = Targets.find(CallerN);
              if (It != Targets.end()) {
                errs() << "TARGET MAPPING " << CallerN << " -> " << CalleeN << '\n';
                unsigned GroupId = It->second;
                TargetsInCallee[CalleeN] = GroupId;

                if (SameSCC) {
                  auto &Offsets = OffsetMaps.at(GroupId);
                  Offsets[CalleeN] = Offsets.at(CallerN);
                } else {
                  assert(OffsetMaps.count(GroupId));
                  assert(OffsetMaps.at(GroupId).count(CalleeN) &&
                      "Offset not found for callee target");
                }
              }
            }
            CalleeRefs =
                Refs.getCalleeTracker(TargetsInCallee, CalleeCallerMapping);
          } else {
            // CalleeG == CallerG
            // no need to propagate if stays in the same dsgraph
            TargetsInCallee = Targets;
            CalleeRefs = Refs;
          }

          if (shouldCloneFunction(Callee, CalleeRefs)) {
            errs() << "--- need to clone callee\n";
            auto *CloneFnTy = computeCloneFnTy(CS, Refs);
            errs() << "--- inferred new callee type after cloning: "
                   << *CloneFnTy << '\n';

            Function *Clone;
            //
            // FIXME: the way we identify reusable clones is broken
            //
            // maybe we have a clone for this use already
            const auto CloneIt =
                CloneCache.find({TargetsInCallee, getSrcFunc(Callee)});
            if (CloneIt != CloneCache.end())
              Clone = CloneIt->second;
            else {
              errs() << "--- cloning " << Callee->getName() << '\n';
              Clone = cloneFunctionWithNewType(Callee, CloneFnTy,
                                               TargetsInCallee, TripleMap);
              errs() << "-- cloned function: " << Clone->getName() << '\n';
              // we haven't processed this new clone
              Worklist.push_back({Clone, CalleeRefs});
            }

            //
            // compute the list of arguments with which we call the clone
            //
            std::vector<Value *> NewArgs;
            unsigned i = 0;
            for (Value *Arg : CS.args()) {
              if (Targets.count(getNodeForValue(Arg, F).getNode())) {
                errs() << "Looking up triple for " << Arg << "\n";
                auto &Triple = TripleMap.at(Arg);
                assert(Triple.Base);
                NewArgs.push_back(Triple.Base);
                NewArgs.push_back(Triple.Idx);
                NewArgs.push_back(Triple.Size);
                i += 3;
              } else {
                auto *ParamTy = CloneFnTy->params()[i++];
                if (Arg->getType() != ParamTy)
                  Arg = CastInst::CreatePointerCast(Arg, ParamTy,
                                                    Arg->getName(), &*I);
                NewArgs.push_back(Arg);
              }
            }

            //
            // if the original function call returns pointer to Target
            // we need to return by reference
            //
            AllocaInst *BaseAddr, *IdxAddr, *SizeAddr;
            bool ReturnsTarget = Targets.count(
                getNodeForValue(CS.getInstruction(), F).getNode());
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
            }

            //
            // call the clone
            //
            assert(Clone->getFunctionType()->getNumParams() == NewArgs.size() &&
                   "mismatch of args in new call");
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
              I->mutateType(NewCall->getType());
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
              errs() << "Setting triple for " << &*I << '\n';
              TripleMap[&*I] = {Base, Idx, Size};
              removeValue(&*I);
            }
            updateValue(&*I, NewCall);
          }
        } // end of handling function call

        //
        if (auto *RI = dyn_cast<ReturnInst>(&*I)) {
          errs() << "-- handling return\n";

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

            auto &Triple = TripleMap.at(RetVal);
            auto SizeAddr = F->getArgumentList().rbegin(),
                 IdxAddr = std::next(SizeAddr), BaseAddr = std::next(IdxAddr);

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
          errs() << "--- handling alloca\n";

          //
          // We need to fix this allocation if it's a target (change size) or it
          // points to a target (pointer conv.)
          //

          if (Targets.count(N)) {
            unsigned GroupId = Targets.at(N);
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

          unsigned SizeNeeded = computeSizeWithTripleConv(NH, Targets);

          auto *TyAfterConv =
              cast<PointerType>(getNewType(AI, NH, Refs))->getElementType();
          //auto *TyAfterConv = getNewType(NH.getNode(), Refs, AI->getAllocatedType(), NH.getOffset());
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
                TD->getTypeAllocSize(NewAllocatedType) == SizeNeeded) {
              // case 1
              // this is the easy one, we just change the type being allocated
              NewAlloca =
                  new AllocaInst(TyAfterConv, AI->getArraySize(), "", AI);
            } else {
              errs() << *AI << ", " << *AllocatedType << ", " << *NewAllocatedType << '\n';
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
            updateValue(AI, NewAlloca);
            removeValue(AI);
          }
        } // end of handling alloca

        if (auto *ICMP = dyn_cast<ICmpInst>(&*I)) {
          errs() << "--- handling icmp\n";
          auto *Ty = ICMP->getOperand(0)->getType();

          for (unsigned i = 0, e = ICMP->getNumOperands(); i != e; i++) {
            auto *OrigOp = ICMP->getOperand(i);
            auto NH = getNodeForValue(OrigOp, F);
            auto *N = NH.getNode();
            if (Targets.count(N)) {
              unsigned GroupId = Targets.at(N);
              unsigned Offset =
                  *OffsetMaps.at(GroupId).at(N).begin() + NH.getOffset();
              assert(TripleMap.count(OrigOp));
              auto &Triple = TripleMap.at(OrigOp);
              assert(Transforms.count(GroupId));
              auto *Addr =
                  Transforms.at(GroupId)->ComputeAddress(Triple, Offset, ICMP);
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
LayoutTuner::computeCloneFnTy(CallSite CS,
                              const RefSetTracker<TargetMapTy> &Refs) {
  errs() << "----- computing clone fn ty\n";
  errs() << "~~~~~~ " << *CS.getInstruction() << '\n';

  Function *Caller = CS.getParent()->getParent(),
           *Callee = CS.getCalledFunction();
  assert(Callee);

  errs() << "---- getting function type for function " << Callee->getName()
         << '\n';
  auto *OrigTy = Callee->getFunctionType();
  std::vector<Type *> ArgTypes;

  errs() << "--- deducing input args for clone\n";
  unsigned NumArgs = OrigTy->getNumParams();
  for (unsigned i = 0; i < NumArgs; i++) {
    Value *Arg = CS.getArgument(i);
    auto *ArgN = getNodeForValue(Arg, Caller).getNode();
    if (Refs.getTargets().count(ArgN)) {
      // expand the pointer to target into triple
      ArgTypes.push_back(Int8PtrTy);
      ArgTypes.push_back(Int64Ty);
      ArgTypes.push_back(Int64Ty);
    } else {
      errs() << "ARG: " << *getOrigValue(Arg) << '\n';
      ArgTypes.push_back(getNewType(const_cast<Value *>(getOrigValue(Arg)),
                                    getNodeForValue(Arg, Caller), Refs));
    }
  }

  auto NH = getNodeForValue(CS.getInstruction(), Caller);

  errs() << "---- deducing return type for clone\n";
  Type *RetTy;
  if (Refs.getTargets().count(NH.getNode())) {
    // in case the function returns a pointer to target
    // we append 3 extra return arguments to set the triple begin "returned"
    RetTy = VoidTy;
    ArgTypes.push_back(PointerType::getUnqual(Int8PtrTy));
    ArgTypes.push_back(PointerType::getUnqual(Int64Ty));
    ArgTypes.push_back(PointerType::getUnqual(Int64Ty));
  } else if (OrigTy->getReturnType() != VoidTy) {
    RetTy = getNewType(CS.getInstruction(), NH, Refs);
  } else
    RetTy = VoidTy;

  return FunctionType::get(RetTy, ArgTypes, false /*isVarArg*/);
}

// TODO maybe we should cache the computation of ReachableNodes
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
  getDSGraph(F)
      ->getCallSiteForArguments(*getSrcFunc(F))
      .getRetVal()
      .getNode()
      ->markReachableNodes(ReachableNodes);

  for (auto *N : ReachableNodes)
    if (Refs.getTargets().count(N) || Refs.refersTargets(N))
      return true;

  return false;
}

// TODO:
// rewrite this to incorporate ::safeToTransform to do legality-check/preprocessing in a single pass
std::set<const DSNode *> LayoutTuner::findTransformableNodes(const Module &M) {
  std::set<const DSNode *> SafeToTransform;

  auto &TypeChecker = getAnalysis<TypeSafety<EquivBUDataStructures>>();

  auto &CG = DSA->getCallGraph();

  std::set<const Function *> Processed;
  for (auto &F : M) {
    if (F.empty())
      continue;
    auto *SccLeader = CG.sccLeader(&F);

    if (!Processed.insert(SccLeader).second)
      continue;

    auto *Graph = DSA->getDSGraph(*SccLeader);

    std::set<const DSNode *> Candidates;
    for (auto N = Graph->node_begin(), NE = Graph->node_end(); N != NE; ++N)
      if (!N->isGlobalNode() && N->isHeapNode() && TypeChecker.isTypeSafe(&*N))
        Candidates.insert(&*N);

    SafeToTransform.insert(Candidates.begin(), Candidates.end());
  }

  return SafeToTransform;
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

  std::error_code EC;
  tool_output_file Out(OutputFilename, EC, sys::fs::F_None);
  if (EC) {
    errs() << EC.message() << '\n';
    return 1;
  }

  legacy::PassManager Passes;
  Passes.add(new LayoutTuner());
  Passes.add(createBitcodeWriterPass(Out.os(), true));
  Passes.add(createVerifierPass());
  Passes.run(*M);

  Out.keep();
}
