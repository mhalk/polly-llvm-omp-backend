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

// #include "clang/CodeGen/CGOpenMPRuntime.h"
#include "polly/CodeGen/LoopGenerators.h"
#include "polly/CodeGen/LoopGeneratorsLLVM.h"
#include "polly/ScopDetection.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define SOME_GLOBAL_TID 1338

// using namespace clang;
// using namespace CodeGen;

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

  printf("LLVM-IR createParallelLoop used.\n");

  AllocaInst *Struct = storeValuesIntoStruct(UsedValues);
  BasicBlock::iterator BeforeLoop = Builder.GetInsertPoint();
  Value *IV = createSubFn(Stride, Struct, UsedValues, Map, &SubFn);
  *LoopBody = Builder.GetInsertPoint();
  Builder.SetInsertPoint(&*BeforeLoop);

  Value *SubFnParam = Builder.CreateBitCast(Struct, Builder.getInt8PtrTy(),
                                            "polly.par.userContext");

  // Add one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UB = Builder.CreateAdd(UB, ConstantInt::get(LongType, 1));

  // Tell the runtime we start a parallel loop
  createCallSpawnThreads(SubFn, SubFnParam, LB, UB, Stride);
  Builder.CreateCall(SubFn, SubFnParam);
  createCallJoinThreads();

  // Mark the end of the lifetime for the parameter struct.
  Type *Ty = Struct->getType();
  ConstantInt *SizeOf = Builder.getInt64(DL.getTypeAllocSize(Ty));
  Builder.CreateLifetimeEnd(Struct, SizeOf);

  return IV;
}

void ParallelLoopGeneratorLLVM::createCallSpawnThreads(Value *SubFn,
                                                   Value *SubFnParam, Value *LB,
                                                   Value *UB, Value *Stride) {
  // const std::string Name = "GOMP_parallel_loop_runtime_start";

  const std::string Name = "__kmpc_for_static_init_4";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    // TODO: Problem getting type of ident_t
    // in LLVM IR:
    // ident_t = type { i32, i32, i32, i32, i8* }
    Type loc = ident_t->getType();

    Type *Params[] = {PointerType::getUnqual(loc),
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

  Value *NumberOfThreads = Builder.getInt32(PollyNumThreads);
  Value *Args[] = { nullptr,
                    SOME_GLOBAL_TID,
                    34, // Static scheduling
                    nullptr,
                    LB, UB, Stride, 1, 1};

  Builder.CreateCall(F, Args);
}

Value *ParallelLoopGeneratorLLVM::createCallGetWorkItem(Value *LBPtr,
                                                    Value *UBPtr) {
  const std::string Name = "GOMP_loop_runtime_next";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {LongType->getPointerTo(), LongType->getPointerTo()};
    FunctionType *Ty = FunctionType::get(Builder.getInt8Ty(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {LBPtr, UBPtr};
  Value *Return = Builder.CreateCall(F, Args);
  Return = Builder.CreateICmpNE(
      Return, Builder.CreateZExt(Builder.getFalse(), Return->getType()));
  return Return;
}

void ParallelLoopGeneratorLLVM::createCallJoinThreads() {
  const std::string Name = "__kmpc_for_static_fini";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type loc = type { i32, i32, i32, i32, i8* };

    struct loc =

    Type *Params[] = {PointerType::getUnqual(loc),
                      Builder.getInt32Ty()};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {nullptr, SOME_GLOBAL_TID});
}

void ParallelLoopGeneratorLLVM::createCallCleanupThread() {
  const std::string Name = "GOMP_loop_end_nowait";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {});
}

Value *ParallelLoopGeneratorLLVM::createSubFn(Value *Stride, AllocaInst *StructData,
                                          SetVector<Value *> Data,
                                          ValueMapT &Map, Function **SubFnPtr) {
  BasicBlock *PrevBB, *HeaderBB, *ExitBB, *CheckNextBB, *PreHeaderBB, *AfterBB;
  Value *LBPtr, *UBPtr, *UserContext, *Ret1, *HasNextSchedule, *LB, *UB, *IV;
  Function *SubFn = createSubFnDefinition();
  LLVMContext &Context = SubFn->getContext();

  // Store the previous basic block.
  PrevBB = Builder.GetInsertBlock();

  // Create basic blocks.
  HeaderBB = BasicBlock::Create(Context, "polly.par.setup", SubFn);
  ExitBB = BasicBlock::Create(Context, "polly.par.exit", SubFn);
  CheckNextBB = BasicBlock::Create(Context, "polly.par.checkNext", SubFn);
  PreHeaderBB = BasicBlock::Create(Context, "polly.par.loadIVBounds", SubFn);

  DT.addNewBlock(HeaderBB, PrevBB);
  DT.addNewBlock(ExitBB, HeaderBB);
  DT.addNewBlock(CheckNextBB, HeaderBB);
  DT.addNewBlock(PreHeaderBB, HeaderBB);

  // Fill up basic block HeaderBB.
  Builder.SetInsertPoint(HeaderBB);
  LBPtr = Builder.CreateAlloca(LongType, nullptr, "polly.par.LBPtr");
  UBPtr = Builder.CreateAlloca(LongType, nullptr, "polly.par.UBPtr");
  UserContext = Builder.CreateBitCast(
      &*SubFn->arg_begin(), StructData->getType(), "polly.par.userContext");

  extractValuesFromStruct(Data, StructData->getAllocatedType(), UserContext,
                          Map);
  Builder.CreateBr(CheckNextBB);

  // Add code to check if another set of iterations will be executed.
  Builder.SetInsertPoint(CheckNextBB);
  Ret1 = createCallGetWorkItem(LBPtr, UBPtr);
  HasNextSchedule = Builder.CreateTrunc(Ret1, Builder.getInt1Ty(),
                                        "polly.par.hasNextScheduleBlock");
  Builder.CreateCondBr(HasNextSchedule, PreHeaderBB, ExitBB);

  // Add code to load the iv bounds for this set of iterations.
  Builder.SetInsertPoint(PreHeaderBB);
  LB = Builder.CreateLoad(LBPtr, "polly.par.LB");
  UB = Builder.CreateLoad(UBPtr, "polly.par.UB");

  // Subtract one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UB = Builder.CreateSub(UB, ConstantInt::get(LongType, 1),
                         "polly.par.UBAdjusted");

  Builder.CreateBr(CheckNextBB);
  Builder.SetInsertPoint(&*--Builder.GetInsertPoint());
  IV = createLoop(LB, UB, Stride, Builder, P, LI, DT, AfterBB,
                  ICmpInst::ICMP_SLE, nullptr, true, /* UseGuard */ false);

  BasicBlock::iterator LoopBody = Builder.GetInsertPoint();

  // Add code to terminate this subfunction.
  Builder.SetInsertPoint(ExitBB);
  createCallCleanupThread();
  Builder.CreateRetVoid();

  Builder.SetInsertPoint(&*LoopBody);
  *SubFnPtr = SubFn;

  return IV;
}
