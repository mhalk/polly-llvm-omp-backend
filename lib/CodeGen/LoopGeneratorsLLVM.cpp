//===------ LoopGenerators.cpp -  IR helper to create loops ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create scalar and parallel loops as LLVM-IR.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/LoopGenerators.h"
#include "polly/CodeGen/LoopGeneratorsLLVM.h"
#include "polly/ScopDetection.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
// #include "clang/CodeGen/CGOpenMPRuntime.h"

using namespace llvm;
using namespace polly;

static cl::opt<int>
    PollyNumThreads("polly-num-threads-llvm",
                    cl::desc("Number of threads to use (0 = auto)"), cl::Hidden,
                    cl::init(0));

/*
Value *ParallelLoopGeneratorLLVM::createLoop(Value *LB, Value *UB, Value *Stride,
                         PollyIRBuilder &Builder, Pass *P, LoopInfo &LI,
                         DominatorTree &DT, BasicBlock *&ExitBB,
                         ICmpInst::Predicate Predicate,
                         ScopAnnotator *Annotator, bool Parallel,
                         bool UseGuard) {
  Function *F = Builder.GetInsertBlock()->getParent();
  LLVMContext &Context = F->getContext();

  assert(LB->getType() == UB->getType() && "Types of loop bounds do not match");
  IntegerType *LoopIVType = dyn_cast<IntegerType>(UB->getType());
  assert(LoopIVType && "UB is not integer?");

  BasicBlock *BeforeBB = Builder.GetInsertBlock();
  BasicBlock *GuardBB =
      UseGuard ? BasicBlock::Create(Context, "polly.loop_if", F) : nullptr;
  BasicBlock *HeaderBB = BasicBlock::Create(Context, "polly.loop_header", F);
  BasicBlock *PreHeaderBB =
      BasicBlock::Create(Context, "polly.loop_preheader", F);

  // Update LoopInfo
  Loop *OuterLoop = LI.getLoopFor(BeforeBB);
  Loop *NewLoop = new Loop();

  if (OuterLoop)
    OuterLoop->addChildLoop(NewLoop);
  else
    LI.addTopLevelLoop(NewLoop);

  if (OuterLoop) {
    if (GuardBB)
      OuterLoop->addBasicBlockToLoop(GuardBB, LI);
    OuterLoop->addBasicBlockToLoop(PreHeaderBB, LI);
  }

  NewLoop->addBasicBlockToLoop(HeaderBB, LI);

  // Notify the annotator (if present) that we have a new loop, but only
  // after the header block is set.
  if (Annotator)
    Annotator->pushLoop(NewLoop, Parallel);

  // ExitBB
  ExitBB = SplitBlock(BeforeBB, &*Builder.GetInsertPoint(), &DT, &LI);
  ExitBB->setName("polly.loop_exit");

  // BeforeBB
  if (GuardBB) {
    BeforeBB->getTerminator()->setSuccessor(0, GuardBB);
    DT.addNewBlock(GuardBB, BeforeBB);

    // GuardBB
    Builder.SetInsertPoint(GuardBB);
    Value *LoopGuard;
    LoopGuard = Builder.CreateICmp(Predicate, LB, UB);
    LoopGuard->setName("polly.loop_guard");
    Builder.CreateCondBr(LoopGuard, PreHeaderBB, ExitBB);
    DT.addNewBlock(PreHeaderBB, GuardBB);
  } else {
    BeforeBB->getTerminator()->setSuccessor(0, PreHeaderBB);
    DT.addNewBlock(PreHeaderBB, BeforeBB);
  }

  // PreHeaderBB
  Builder.SetInsertPoint(PreHeaderBB);
  Builder.CreateBr(HeaderBB);

  // HeaderBB
  DT.addNewBlock(HeaderBB, PreHeaderBB);
  Builder.SetInsertPoint(HeaderBB);
  PHINode *IV = Builder.CreatePHI(LoopIVType, 2, "polly.indvar");
  IV->addIncoming(LB, PreHeaderBB);
  if (Stride->getType() != LoopIVType) {
    printf("OMG");
    Stride->dump();
    LoopIVType->dump();
    printf("WTF");
    Stride = Builder.CreateZExtOrBitCast(Stride, LoopIVType);
  }
  Value *IncrementedIV = Builder.CreateNSWAdd(IV, Stride, "polly.indvar_next");
  Value *LoopCondition;
  UB->dump();
  UB = Builder.CreateSub(UB, Stride, "polly.adjust_ub");
  LoopCondition = Builder.CreateICmp(Predicate, IV, UB);
  LoopCondition->setName("polly.loop_cond");

  LoopCondition->dump();
  UB->dump();
  Stride->dump();
  LoopIVType->dump();

  // Create the loop latch and annotate it as such.
  BranchInst *B = Builder.CreateCondBr(LoopCondition, HeaderBB, ExitBB);
  if (Annotator)
    Annotator->annotateLoopLatch(B, NewLoop, Parallel);

  IV->addIncoming(IncrementedIV, HeaderBB);
  if (GuardBB)
    DT.changeImmediateDominator(ExitBB, GuardBB);
  else
    DT.changeImmediateDominator(ExitBB, HeaderBB);

  // The loop body should be added here.
  Builder.SetInsertPoint(HeaderBB->getFirstNonPHI());
  return IV;
}
*/


Value *ParallelLoopGeneratorLLVM::createParallelLoop(
    Value *LB, Value *UB, Value *Stride, SetVector<Value *> &UsedValues,
    ValueMapT &Map, BasicBlock::iterator *LoopBody) {
  Function *SubFn;
  StructType *identTy = M->getTypeByName("ident_t");

  printf("LLVM-IR createParallelLoop used.\n");

  // If the ident_t StructType is not available, declare it.
  // in LLVM-IR: ident_t = type { i32, i32, i32, i32, i8* }
  if (!identTy) {
    Type *loc_members[] = {Builder.getInt32Ty(), Builder.getInt32Ty(),
                           Builder.getInt32Ty(), Builder.getInt32Ty(),
                           Builder.getInt8PtrTy() };

    identTy = StructType::create(M->getContext(), loc_members, "ident_t", false);
  }

  printf("LLVM-IR createParallelLoop:\t Pos 02.1\n");

  int size = 23;
  auto arrayType = llvm::ArrayType::get(Builder.getInt8Ty(), size);
  printf("LLVM-IR createParallelLoop:\t Pos 02,21\n");
  // Global Variable Definitions
  GlobalVariable* theString = new GlobalVariable(*M, arrayType, true,
                          GlobalValue::PrivateLinkage, 0, ".strIdent");
  theString->setAlignment(1);

  GlobalVariable *dummy_src_loc = new GlobalVariable(*M, identTy, true,
  GlobalValue::PrivateLinkage, nullptr, "dummy.src_loc");
  dummy_src_loc->setAlignment(8);

  printf("LLVM-IR createParallelLoop:\t Pos 02,22\n");

  // Constant Definitions
  Constant *locInit_str = ConstantDataArray::getString(M->getContext(),
  ";unknown;unknown;0;0;;", true);

  theString->setInitializer(locInit_str);

  printf("LLVM-IR createParallelLoop:\t Pos 02,24\n");

  printf("LLVM-IR createParallelLoop:\t Pos 02,3\n");
  Value *aStringGEP = Builder.CreateInBoundsGEP(arrayType, theString,
    {Builder.getInt32(0), Builder.getInt32(0)});

  printf("LLVM-IR createParallelLoop:\t Pos 03\n");

  Constant *cZero = Builder.getInt32(0);
  Constant *locInit_struct = ConstantStruct::get(identTy,
    {cZero, cZero, cZero, cZero, (Constant*) aStringGEP});

  dummy_src_loc->setInitializer(locInit_struct);

  printf("LLVM-IR createParallelLoop:\t Pos 03,1\n");

  BasicBlock::iterator BeforeLoop = Builder.GetInsertPoint();
  Value *IV = createSubFn(LB, UB, Stride, &SubFn, dummy_src_loc);
  *LoopBody = Builder.GetInsertPoint();
  Builder.SetInsertPoint(&*BeforeLoop);

  // UB->dump();
  // LongType->dump();

  // Add one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UB = Builder.CreateAdd(UB, ConstantInt::get(LongType, 1));

  printf("LLVM-IR createParallelLoop:\t Pos 04\n");

  // Tell the runtime we start a parallel loop
  createCallSpawnThreads(dummy_src_loc, SubFn);

  printf("LLVM-IR createParallelLoop:\t Pos 05\n");

  // TODO: Check if neccessary!
  //Builder.CreateCall(SubFn, {vZero, vZero});

  freopen("/home/mhalk/ba/dump/moddump.ll", "w", stderr);
  M->dump();
  freopen("/dev/tty", "w", stderr);

  printf("LLVM-IR createParallelLoop:\t Pos 06\n");

  freopen("/home/mhalk/ba/dump/moddump_endOfcreateParLoop.ll", "w", stderr);
  M->dump();
  freopen("/dev/tty", "w", stderr);

  return IV;
}

void ParallelLoopGeneratorLLVM::createCallSpawnThreads(Value *_loc,
                                                       Value *_microtask) {
  printf("LLVM-IR createCallSpawnThreads:\tEntry\n");

  // const std::string Name = "GOMP_parallel_loop_runtime_start";
  const std::string Name = "__kmpc_fork_call";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    StructType *identTy = M->getTypeByName("ident_t");
    Type *Kmpc_MicroTy = M->getTypeByName("kmpc_micro");

    if (!Kmpc_MicroTy) {
       // Build void (*kmpc_micro)(kmp_int32 *global_tid, kmp_int32 *bound_tid,...)
       Type *MicroParams[] = {Builder.getInt32Ty()->getPointerTo(),
                              Builder.getInt32Ty()->getPointerTo()};

       Kmpc_MicroTy = FunctionType::get(Builder.getVoidTy(), MicroParams, true);
    }

    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {identTy->getPointerTo(),
                      Builder.getInt32Ty(),
                      Kmpc_MicroTy->getPointerTo()};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, true);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  printf("LLVM-IR createCallSpawnThreads:\tFunc defined: Pos 02\n");

  Value *Args[] = {_loc, Builder.getInt32(0), _microtask};

  printf("LLVM-IR createCallSpawnThreads:\tFunc defined: Pos 03\n");

  Builder.CreateCall(F, Args);

  printf("LLVM-IR createCallSpawnThreads:\tFunc defined: Pos 04\n");

  freopen("/home/mhalk/ba/dump/moddump.ll", "w", stderr);
  M->dump();
  freopen("/dev/tty", "w", stderr);

  printf("LLVM-IR createCallSpawnThreads:\tExit\n");
}

void ParallelLoopGeneratorLLVM::createCallGetWorkItem(Value *loc,
                                                   Value *global_tid,
                                                   Value *pIsLast, Value *pLB,
                                                   Value *pUB, Value *pStride) {
  printf("LLVM-IR createCallGetWorkItem:\tEntry\n");

  const std::string Name = "__kmpc_for_static_init_4";
  Function *F = M->getFunction(Name);
  StructType *identTy = M->getTypeByName("ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    // StructType *loc = StructType::get(M->getContext(), loc_members, false);

    Type *Params[] = {identTy->getPointerTo(), Builder.getInt32Ty(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty()->getPointerTo(),
                      Builder.getInt32Ty()->getPointerTo(),
                      Builder.getInt32Ty()->getPointerTo(),
                      Builder.getInt32Ty()->getPointerTo(),
                      Builder.getInt32Ty(), Builder.getInt32Ty()};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  printf("LLVM-IR createCallGetWorkItem:\tFunc defined: Pos 05\n");

  // Value *NumberOfThreads = Builder.getInt32(PollyNumThreads);
  Value *Args[] = { loc,
                    global_tid,
                    Builder.getInt32(34), // Static scheduling
                    pIsLast,
                    pLB,
                    pUB,
                    pStride,
                    Builder.getInt32(1),
                    ConstantInt::get(Builder.getInt32Ty(), 1, true) };

  printf("LLVM-IR createCallGetWorkItem:\tFunc defined: Pos 07\n");

  Builder.CreateCall(F, Args);
  printf("LLVM-IR createCallGetWorkItem:\tExit\n");
}

void ParallelLoopGeneratorLLVM::createCallJoinThreads() {
  /*
  // const std::string Name = "GOMP_parallel_end";

  printf("LLVM-IR createCallJoinThreads:\tFunc defined: EXIT\n");
  */
  printf("createCallJoinThreads:\tCALLED in LLVM OpenMP mode!\n");
}

void ParallelLoopGeneratorLLVM::createCallCleanupThread(Value* _loc, Value* _id) {
  // const std::string Name = "GOMP_loop_end_nowait";
  printf("LLVM-IR createCallCleanupThread:\tEntry\n");
  const std::string Name = "__kmpc_for_static_fini";
  Function *F = M->getFunction(Name);
  StructType *identTy = M->getTypeByName("ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {identTy->getPointerTo(), Builder.getInt32Ty()};
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  printf("LLVM-IR createCallCleanupThread:\tFunc defined: Pos 01\n");

  // Value *NumberOfThreads = Builder.getInt32(PollyNumThreads);
  Value *Args[] = { _loc, _id };

  printf("LLVM-IR createCallCleanupThread:\tFunc defined: Pos 02\n");

  Builder.CreateCall(F, Args);
}

Value *ParallelLoopGeneratorLLVM::createSubFn(Value *LB, Value *UB,
                  Value *Stride, Function **SubFnPtr, Value *Location) {
  BasicBlock *PrevBB, *HeaderBB, *ExitBB, *CheckNextBB, *PreHeaderBB, *AfterBB;
  Value *vLB, *vUB, *vStride, *LBPtr, *UBPtr, *IDPtr,
    *ID, *IV, *pLB, *pUB, *pIsLast, *pStride;
  printf("LLVM-IR createSubFn:\tEntry\n");
  Function *SubFn = createSubFnDefinition();
  LLVMContext &Context = SubFn->getContext();

  // Store the previous basic block.
  PrevBB = Builder.GetInsertBlock();

  printf("LLVM-IR createSubFn:\t 01\n");

  // Create basic blocks.
  HeaderBB = BasicBlock::Create(Context, "polly.par.setup", SubFn);
  ExitBB = BasicBlock::Create(Context, "polly.par.exit", SubFn);
  CheckNextBB = BasicBlock::Create(Context, "polly.par.checkNext", SubFn);
  PreHeaderBB = BasicBlock::Create(Context, "polly.par.loadIVBounds", SubFn);

  DT.addNewBlock(HeaderBB, PrevBB);
  DT.addNewBlock(ExitBB, HeaderBB);
  DT.addNewBlock(CheckNextBB, HeaderBB);
  DT.addNewBlock(PreHeaderBB, HeaderBB); // ORIGINAL
  // DUMMIES
  // DT.addNewBlock(CheckNextBB, PreHeaderBB);

  printf("LLVM-IR createSubFn:\t 02\n");

  // Fill up basic block HeaderBB.
  Builder.SetInsertPoint(HeaderBB);

  IDPtr = &*SubFn->arg_begin();

  LBPtr = Builder.CreateAlloca(LongType, nullptr, "polly.par.LBPtr");
  UBPtr = Builder.CreateAlloca(LongType, nullptr, "polly.par.UBPtr");
  pIsLast = Builder.CreateAlloca(Builder.getInt32Ty(), nullptr, "polly.par.pIsLast");
  pStride = Builder.CreateAlloca(Builder.getInt32Ty(), nullptr, "polly.par.pStride");

  pLB = Builder.CreateBitCast(LBPtr, Builder.getInt32Ty()->getPointerTo(), "polly.par.LB32");
  pUB = Builder.CreateBitCast(UBPtr, Builder.getInt32Ty()->getPointerTo(), "polly.par.UB32");

  Builder.CreateAlignedLoad(LBPtr, 4, "polly.par.LB");
  Builder.CreateAlignedLoad(UBPtr, 4, "polly.par.UB");
  ID = Builder.CreateAlignedLoad(IDPtr, 4, "polly.par.global_tid");

  // *vUB = ConstantInt::get(Builder.getInt32Ty(), 1, true);
  // *vStride = ConstantInt::get(Builder.getInt32Ty(), 42, true);

  printf("LLVM-IR createSubFn:\t 02.2\n");

  int constIntValue_LB = -1, constIntValue_UB = -2, constIntValue_Stride = -3;

  if (ConstantInt* CI = dyn_cast<ConstantInt>(LB)) {
    if (CI->getBitWidth() <= 32) {
      constIntValue_LB = CI->getSExtValue();
      vLB = vUB = ConstantInt::get(LongType, constIntValue_LB, true);
    }
  }

  if (ConstantInt* CI = dyn_cast<ConstantInt>(UB)) {
    if (CI->getBitWidth() <= 32) {
      constIntValue_UB = CI->getSExtValue();
      // Increment upper bound by one -- difference between KMPC / GNU lib
      vUB = ConstantInt::get(LongType, constIntValue_UB, true);
    }
  }

  if (ConstantInt* CI = dyn_cast<ConstantInt>(Stride)) {
    if (CI->getBitWidth() <= 32) {
      constIntValue_Stride = CI->getSExtValue();
      vStride = ConstantInt::get(LongType, constIntValue_Stride, true);
    }
  }

  printf("### -- LB: %d -- UB: %d -- Stride: %d ###\n",
  constIntValue_LB, constIntValue_UB, constIntValue_Stride);

  Builder.CreateAlignedStore(vLB, pLB, 4);
  Builder.CreateAlignedStore(vUB, pUB, 4);
  Builder.CreateAlignedStore(Builder.getInt32(0), pIsLast, 4);
  Builder.CreateAlignedStore(vStride, pStride, 4);

  printf("LLVM-IR createSubFn:\t 02.4\n");
  // Subtract one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.

  freopen("/home/mhalk/ba/dump/moddump.ll", "w", stderr);
  M->dump();
  freopen("/dev/tty", "w", stderr);

  // insert __kmpc_for_static_init_4 HERE
  createCallGetWorkItem(Location, ID, pIsLast, pLB, pUB, pStride);

  Builder.SetInsertPoint(HeaderBB);

  LB = Builder.CreateAlignedLoad(LBPtr, 4, "polly.indvar.init");
  UB = Builder.CreateAlignedLoad(UBPtr, 4, "polly.indvar.UB");
  //UB = Builder.CreateAdd(UB, ConstantInt::get(LongType, -1),
  //                       "polly.indvar.UBAdjusted");

  Value *UB32 = Builder.CreateAlignedLoad(pUB, 4);
  Value *selectCond = Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT, UB32, vUB, "polly.UB_slt_42");
  Value *cond = Builder.CreateSelect(selectCond, UB32, vUB);
  Builder.CreateAlignedStore(cond, pUB, 4);

  Value *lowBound = Builder.CreateAlignedLoad(pLB, 4);
  Value *hasIteration = Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT, lowBound, cond, "polly.hasIteration");
  Builder.CreateCondBr(hasIteration, PreHeaderBB, ExitBB);

  //Builder.CreateBr(CheckNextBB);

  // Add code to check if another set of iterations will be executed.
  Builder.SetInsertPoint(CheckNextBB);

  // UB = Builder.CreateSub(UB, ConstantInt::get(LongType, 1), "polly.par.UBAdjusted");
  Builder.CreateBr(ExitBB);

  // Value *loc, Value *global_tid,
  // Value *pIsLast, Value *pLB,
  // Value *pUB, Value *pStride

  printf("LLVM-IR createSubFn:\t 03\n");

  printf("LLVM-IR createSubFn:\t 04\n");

  // Builder.SetInsertPoint(&*--Builder.GetInsertPoint());
  Builder.SetInsertPoint(PreHeaderBB);

  UB = Builder.CreateSub(UB, ConstantInt::get(LongType, 1),
                         "polly.par.UBAdjusted");

  Builder.CreateBr(CheckNextBB);
  Builder.SetInsertPoint(&*--Builder.GetInsertPoint());

  freopen("/home/mhalk/ba/dump/moddump.ll", "w", stderr);
  M->dump();
  freopen("/dev/tty", "w", stderr);

  printf("LLVM-IR createSubFn:\t 04.2\n");
  IV = createLoop(LB, UB, Stride, Builder, P, LI, DT, AfterBB,
                  ICmpInst::ICMP_SLE, nullptr, true, false);
  printf("LLVM-IR createSubFn:\t 04.3\n");
  BasicBlock::iterator LoopBody = Builder.GetInsertPoint();

  printf("LLVM-IR createSubFn:\t 05\n");

  // Add code to terminate this subfunction.
  Builder.SetInsertPoint(ExitBB);
  createCallCleanupThread(Location, ID);
  Builder.CreateRetVoid();

  Builder.SetInsertPoint(&*LoopBody);
  *SubFnPtr = SubFn;

  printf("LLVM-IR createSubFn:\tExit.\n");

  return IV;
}

Function *ParallelLoopGeneratorLLVM::createSubFnDefinition() {
  printf("LLVM-IR createSubFnDefinition:\tEntry\n");
  Function *F = Builder.GetInsertBlock()->getParent();
  Type *MicroParams[] = {Builder.getInt32Ty()->getPointerTo(),
                         Builder.getInt32Ty()->getPointerTo()};
  FunctionType *FT = FunctionType::get(Builder.getVoidTy(), MicroParams, true);
  Function *SubFn = Function::Create(FT, Function::InternalLinkage,
                                     F->getName() + "_polly_subfn", M);

  // Certain backends (e.g., NVPTX) do not support '.'s in function names.
  // Hence, we ensure that all '.'s are replaced by '_'s.
  std::string FunctionName = SubFn->getName();
  std::replace(FunctionName.begin(), FunctionName.end(), '.', '_');
  SubFn->setName(FunctionName);

  // Do not run any polly pass on the new function.
  SubFn->addFnAttr(PollySkipFnAttr);

  // Name function parameters
  Function::arg_iterator AI = SubFn->arg_begin();
  AI->setName("polly.kmpc.global_tid");
  ++AI;
  AI->setName("polly.kmpc.bound_tid");

  printf("LLVM-IR createSubFnDefinition:\tExit\n");

  return SubFn;
}
