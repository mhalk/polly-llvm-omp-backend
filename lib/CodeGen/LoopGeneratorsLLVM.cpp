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

Value *ParallelLoopGeneratorLLVM::createParallelLoop(
    Value *LB, Value *UB, Value *Stride, SetVector<Value *> &UsedValues,
    ValueMapT &Map, BasicBlock::iterator *LoopBody) {
  Function *SubFn;
  StructType *identTy = M->getTypeByName("ident_t");
  // identTy = CGOpenMPRuntime::getIdentTyPointerTy();

  printf("LLVM-IR createParallelLoop used.\n");

  // If the ident_t StructType is not available, declare it.
  // in LLVM-IR: ident_t = type { i32, i32, i32, i32, i8* }
  if (!identTy) {
    Type *loc_members[] = {Builder.getInt32Ty(), Builder.getInt32Ty(),
                           Builder.getInt32Ty(), Builder.getInt32Ty(),
                           Builder.getInt8PtrTy() };

    identTy = StructType::create(M->getContext(), loc_members, "ident_t", false);
  }

  printf("LLVM-IR createParallelLoop:\t Pos 02\n");

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
  Constant *locInit_str = ConstantDataArray::getString(M->getContext(), ";unknown;unknown;0;0;;", true);

  theString->setInitializer(locInit_str);

  printf("LLVM-IR createParallelLoop:\t Pos 02,24\n");

  printf("LLVM-IR createParallelLoop:\t Pos 02,3\n");
  Value *aStringGEP = Builder.CreateInBoundsGEP(arrayType, theString, {Builder.getInt32(0), Builder.getInt32(0)});

  printf("LLVM-IR createParallelLoop:\t Pos 03\n");

  Constant *cZero = ConstantInt::get(Builder.getInt32Ty(), 0, true);
  Constant *locInit_struct = ConstantStruct::get(identTy, {cZero, cZero, cZero, cZero, (Constant*) aStringGEP});

  dummy_src_loc->setInitializer(locInit_struct);

  printf("LLVM-IR createParallelLoop:\t Pos 03,1\n");

  AllocaInst *Struct = storeValuesIntoStruct(UsedValues);
  BasicBlock::iterator BeforeLoop = Builder.GetInsertPoint();
  Value *IV = createSubFn(Stride, Struct, UsedValues, Map, &SubFn, dummy_src_loc);
  *LoopBody = Builder.GetInsertPoint();
  Builder.SetInsertPoint(&*BeforeLoop);

  // Value *SubFnParam = Builder.CreateBitCast(Struct, Builder.getInt8PtrTy(),
  //                                          "polly.par.userContext");



  // Add one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UB = Builder.CreateAdd(UB, ConstantInt::get(LongType, 1));

  printf("LLVM-IR createParallelLoop:\t Pos 04\n");

  // Tell the runtime we start a parallel loop
  createCallSpawnThreads(dummy_src_loc, SubFn);
  // SubFn->dump();

  printf("LLVM-IR createParallelLoop:\t Pos 05\n");

  freopen("/home/mhalk/ba/dump/moddump.ll", "w", stderr);
  M->dump();
  freopen("/dev/tty", "w", stderr);

  // TODO: Check if neccessary!
  //Builder.CreateCall(SubFn, {vZero, vZero});
  // createCallJoinThreads();

  freopen("/home/mhalk/ba/dump/moddump.ll", "w", stderr);
  M->dump();
  freopen("/dev/tty", "w", stderr);

  printf("LLVM-IR createParallelLoop:\t Pos 06\n");
  // SubFn->dump();

  // Mark the end of the lifetime for the parameter struct.
  Type *Ty = Struct->getType();
  ConstantInt *SizeOf = Builder.getInt64(DL.getTypeAllocSize(Ty));
  Builder.CreateLifetimeEnd(Struct, SizeOf);

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


  Value *vZero = ConstantInt::get(Builder.getInt32Ty(), 0, true);
  Value *Args[] = {_loc, vZero, _microtask};

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

  // If the ident_t StructType is not available, declare it.
  // in LLVM-IR: ident_t = type { i32, i32, i32, i32, i8* }
  /*
  if (!identTy) {
    Type *loc_members[] = {Builder.getInt32Ty(), Builder.getInt32Ty(),
                           Builder.getInt32Ty(), Builder.getInt32Ty(),
                           Builder.getInt8PtrTy() };

    identTy = StructType::create(M->getContext(), loc_members, "ident_t", false);
  }
  */

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    // StructType *loc = StructType::get(M->getContext(), loc_members, false);

    Type *Params[] = {identTy->getPointerTo(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty()->getPointerTo(),
                      Builder.getInt32Ty()->getPointerTo(),
                      Builder.getInt32Ty()->getPointerTo(),
                      Builder.getInt32Ty()->getPointerTo(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty()};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  printf("LLVM-IR createCallGetWorkItem:\tFunc defined: Pos 05\n");

  /*
  Value *my_src_loc = Builder.CreateAlloca(identTy, nullptr, "my_src_loc");
  Value *my_isLastIter = Builder.CreateAlloca(Builder.getInt32Ty(), nullptr, "my_isLastIter");
  Value *my_LB = Builder.CreateAlloca(Builder.getInt32Ty(), nullptr, "my_LB");
  Value *my_UB = Builder.CreateAlloca(Builder.getInt32Ty(), nullptr, "my_UB");
  Value *my_Stride = Builder.CreateAlloca(Builder.getInt32Ty(), nullptr, "my_stride");
  Value *my_itest = Builder.CreateAlloca(Builder.getInt32Ty(), nullptr, "my_isLastIter");
  */

  // Value *NumberOfThreads = Builder.getInt32(PollyNumThreads);
  Value *Args[] = { loc,
                    global_tid,
                    ConstantInt::get(Builder.getInt32Ty(), 34, true), // Static scheduling
                    pIsLast,
                    pLB,
                    pUB,
                    pStride,
                    ConstantInt::get(Builder.getInt32Ty(), 1, true),
                    ConstantInt::get(Builder.getInt32Ty(), 1, true) };

  printf("LLVM-IR createCallGetWorkItem:\tFunc defined: Pos 07\n");

  Builder.CreateCall(F, Args);
  printf("LLVM-IR createCallGetWorkItem:\tExit\n");
}

void ParallelLoopGeneratorLLVM::createCallJoinThreads() {
  /*
  // const std::string Name = "GOMP_parallel_end";
  const std::string Name = "__kmpc_for_static_fini";
  Function *F = M->getFunction(Name);
  StructType *identTy = M->getTypeByName("ident_t");

  printf("LLVM-IR createCallJoinThreads:\tFunc not defined: Pos 01\n");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {identTy->getPointerTo(),
                      Builder.getInt32Ty()};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  printf("LLVM-IR createCallJoinThreads:\tFunc defined: Pos 02\n");

  Value *my_src_loc = Builder.CreateAlloca(identTy, nullptr, "my_src_loc1");
  Value *Args[] = { my_src_loc,
                    ConstantInt::get(Builder.getInt32Ty(), SOME_GLOBAL_TID, true)};

  Builder.CreateCall(F, Args);

  printf("LLVM-IR createCallJoinThreads:\tFunc defined: EXIT\n");
  */
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

  F->dump();

  printf("LLVM-IR createCallCleanupThread:\tFunc defined: Pos 05\n");

  // Value *NumberOfThreads = Builder.getInt32(PollyNumThreads);
  Value *Args[] = { _loc, _id };

  printf("LLVM-IR createCallCleanupThread:\tFunc defined: Pos 06\n");

  freopen("/home/mhalk/ba/dump/moddump.ll", "w", stderr);
  M->dump();
  freopen("/dev/tty", "w", stderr);

  printf("LLVM-IR createCallCleanupThread:\tFunc defined: Pos 07\n");

  Builder.CreateCall(F, Args);
}

Value *ParallelLoopGeneratorLLVM::createSubFn(Value *Stride, AllocaInst *StructData,
                                          SetVector<Value *> Data,
                                          ValueMapT &Map, Function **SubFnPtr, Value *Location) {
  BasicBlock *PrevBB, *HeaderBB, *ExitBB, *CheckNextBB, *PreHeaderBB, *AfterBB;
  Value *LBPtr, *UBPtr, *IDPtr, *LB, *UB, *ID, *IV, *pLB, *pUB, *pIsLast, *pStride;
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

  LB = Builder.CreateAlignedLoad(LBPtr, 8, "polly.par.LB");
  UB = Builder.CreateAlignedLoad(UBPtr, 8, "polly.par.UB");
  ID = Builder.CreateAlignedLoad(IDPtr, 8, "polly.par.global_tid");

  Value *vZero = ConstantInt::get(Builder.getInt32Ty(), 0, true);
  Value *vOne = ConstantInt::get(Builder.getInt32Ty(), 1, true);
  Value *vFortyTwo = ConstantInt::get(Builder.getInt32Ty(), 42, true);

  printf("LLVM-IR createSubFn:\t 02.2\n");

  Builder.CreateAlignedStore(vZero, pLB, 8);
  Builder.CreateAlignedStore(vFortyTwo, pUB, 8);
  Builder.CreateAlignedStore(vZero, pIsLast, 4);
  Builder.CreateAlignedStore(vOne, pStride, 4);

  printf("LLVM-IR createSubFn:\t 02.4\n");
  // Subtract one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.

  freopen("/home/mhalk/ba/dump/moddump.ll", "w", stderr);
  M->dump();
  freopen("/dev/tty", "w", stderr);

  // insert __kmpc_for_static_init_4 HERE
  createCallGetWorkItem(Location, ID, pIsLast, pLB, pUB, pStride);

  Builder.SetInsertPoint(HeaderBB);


  LB = Builder.CreateAlignedLoad(LBPtr, 8, "polly.indvar.init");
  UB = Builder.CreateAlignedLoad(UBPtr, 8, "polly.indvar.UB");
  UB = Builder.CreateAdd(UB, ConstantInt::get(LongType, -1),
                         "polly.indvar.UBAdjusted");

  Builder.CreateBr(CheckNextBB);

  // Add code to check if another set of iterations will be executed.
  Builder.SetInsertPoint(CheckNextBB);

  // UB = Builder.CreateSub(UB, ConstantInt::get(LongType, 1),
  //                       "polly.par.UBAdjusted");


  Value *UB32 = Builder.CreateAlignedLoad(pUB, 8);
  Value *selectCond = Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT, UB32, vFortyTwo, "polly.UB_slt_42");
  Value *cond = Builder.CreateSelect(selectCond, UB32, vFortyTwo);
  Builder.CreateAlignedStore(cond, pUB, 8);

  Value *lowBound = Builder.CreateAlignedLoad(pLB, 8);
  Value *hasIteration = Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT, lowBound, cond, "polly.hasIteration");
  Builder.CreateCondBr(hasIteration, PreHeaderBB, ExitBB);

  // Value *loc, Value *global_tid,
  // Value *pIsLast, Value *pLB,
  // Value *pUB, Value *pStride

  printf("LLVM-IR createSubFn:\t 03\n");

  printf("LLVM-IR createSubFn:\t 04\n");

  // Builder.SetInsertPoint(&*--Builder.GetInsertPoint());
  Builder.SetInsertPoint(PreHeaderBB);
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

  Function::arg_iterator AI = SubFn->arg_begin();
  printf("LLVM-IR createSubFnDefinition:\t 01\n");
  AI->setName("polly.global_tid");
  printf("LLVM-IR createSubFnDefinition:\t 02\n");
  ++AI;
  printf("LLVM-IR createSubFnDefinition:\t 03\n");
  AI->setName("polly.bound_tid");
  printf("LLVM-IR createSubFnDefinition:\t 04\n");

  SubFn->dump();

  printf("LLVM-IR createSubFnDefinition:\tExit\n");

  return SubFn;
}
