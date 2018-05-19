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
  AllocaInst *Struct = storeValuesIntoStruct(UsedValues);

  printf("LLVM-IR createParallelLoop used.\n");

  // If the ident_t StructType is not available, declare it.
  // in LLVM-IR: ident_t = type { i32, i32, i32, i32, i8* }
  if (!identTy) {
    Type *loc_members[] = {Builder.getInt32Ty(), Builder.getInt32Ty(),
                           Builder.getInt32Ty(), Builder.getInt32Ty(),
                           Builder.getInt8PtrTy() };

    identTy = StructType::create(M->getContext(), loc_members, "ident_t", false);
  }

  printf("LLVM-IR createParallelLoop:\t Pos 02.21\n");

  int strLen = 23;
  auto arrayType = llvm::ArrayType::get(Builder.getInt8Ty(), strLen);

  // Global Variable Definitions
  GlobalVariable* theString = new GlobalVariable(*M, arrayType, true,
                          GlobalValue::PrivateLinkage, 0, ".strIdent");
  theString->setAlignment(1);

  GlobalVariable *dummy_src_loc = new GlobalVariable(*M, identTy, true,
  GlobalValue::PrivateLinkage, nullptr, "dummy.src_loc");
  dummy_src_loc->setAlignment(8);

  printf("LLVM-IR createParallelLoop:\t Pos 02.22\n");

  // Constant Definitions
  Constant *locInit_str = ConstantDataArray::getString(M->getContext(),
  ";unknown;unknown;0;0;;", true);

  theString->setInitializer(locInit_str);

  Value *aStringGEP = Builder.CreateInBoundsGEP(arrayType, theString,
    {Builder.getInt32(0), Builder.getInt32(0)});

  printf("LLVM-IR createParallelLoop:\t Pos 03\n");

  Constant *cZero = Builder.getInt32(0);
  Constant *locInit_struct = ConstantStruct::get(identTy,
    {cZero, cZero, cZero, cZero, (Constant*) aStringGEP});

  dummy_src_loc->setInitializer(locInit_struct);

  printf("LLVM-IR createParallelLoop:\t Pos 03.1\n");

  int numThreads = (PollyNumThreads <= 0) ? 4 : PollyNumThreads;
  numThreads = 4;

/*

  Value *multiplier = Builder.getInt32(numThreads); // TODO TODO !!! Need Offset ???

  if (ConstantInt* CI = dyn_cast<ConstantInt>(LB)) {
    //if (CI->getBitWidth() <= 64) {
      int constIntValue_LB = CI->getSExtValue();
      LB = ConstantInt::get(Builder.getInt32Ty(), constIntValue_LB, true);
    //}
  } else {
    printf("Truncating LB!\n");
    LB->dump();
    //LB = Builder.CreateTrunc(LB, LongType, "polly.truncLB.vanilla");
    //LB = Builder.CreateMul(LB, multiplier, "polly.par.var_arg.LBx4");
    //LB->dump();
    LB = Builder.CreateTrunc(LB, LongType, "polly.truncLB");
  }

  printf("LLVM-IR createParallelLoop:\tBEFORE: Truncating UB!\n");
  UB->dump();


  if (ConstantInt* CI = dyn_cast<ConstantInt>(UB)) {
    //if (CI->getBitWidth() <= 64) {
      int constIntValue_UB = CI->getSExtValue();
      UB = ConstantInt::get(Builder.getInt32Ty(), constIntValue_UB+1, true);
    //}
  } else {
    // UB = Builder.CreateShl(UB, Builder.getInt32(2), "polly.par.var_arg.UBx4");
    UB->dump();
    UB = Builder.CreateTrunc(UB, LongType, "polly.truncUB.vanilla");
    //UB = Builder.CreateShl(UB, Builder.getInt32(2), "polly.par.var_arg.UBx4");
    UB = Builder.CreateAdd(UB, ConstantInt::get(LongType, numThreads), "polly.truncUB.incr2");
    UB = Builder.CreateMul(UB, multiplier, "polly.par.var_arg.UBx4");
    UB->dump();
    //UB = Builder.CreateAdd(UB, ConstantInt::get(LongType, 10), "polly.truncUB.incr2");
    //UB = Builder.CreateAdd(UB, multiplier, "polly.truncUB.incr");
    //UB = Builder.CreateTrunc(UB, LongType, "polly.truncUB");
  }

  printf("LLVM-IR createParallelLoop:\tAFTER: Truncating UB! NumThreads:\n");
  //UB = Builder.CreateTrunc(UB, Builder.getInt32Ty(), "polly.truncUB");
  //UB = Builder.CreateAdd(UB, ConstantInt::get(LongType, 1), "polly.truncUB.incr");
  UB->dump();

  if (ConstantInt* CI = dyn_cast<ConstantInt>(Stride)) {
    //if (CI->getBitWidth() <= 64) {
      int constIntValue_Stride = CI->getSExtValue();
      Stride = ConstantInt::get(Builder.getInt32Ty(), constIntValue_Stride, true);
    //}
  } else {
    printf("Truncating Stride!\n");
    Stride->dump();
    //Stride = Builder.CreateTrunc(Stride, LongType, "polly.truncStride.vanilla");
    //Stride = Builder.CreateMul(Stride, NumberOfThreadsIncr, "polly.par.var_arg.Stridex4");
    Stride->dump();
    Stride = Builder.CreateTrunc(Stride, LongType, "polly.truncStride");
  }

  */

  //UB = Builder.CreateAdd(UB, ConstantInt::get(LongType, numThreads), "polly.truncUB.incr2");
  //UB = Builder.CreateMul(UB, ConstantInt::get(LongType, numThreads), "polly.par.var_arg.UBx4");

  printf("--- BEFORE createSubFn!\n");
  LB->dump();
  UB->dump();

  BasicBlock::iterator BeforeLoop = Builder.GetInsertPoint();
  Value *IV = createSubFn(Struct, UsedValues, Map, &SubFn, dummy_src_loc);
  *LoopBody = Builder.GetInsertPoint();
  Builder.SetInsertPoint(&*BeforeLoop);

  // TODO : Experimentieren
  Value *SubFnParam = Builder.CreateBitCast(Struct, Builder.getInt8PtrTy(), "polly.par.userContext"); // Original
  // Value *SubFnParam = Builder.CreateBitCast(Struct, Builder.getInt32Ty()->getPointerTo(), "polly.par.userContext"); // kmpc 32bit ?

  printf("LLVM-IR createParallelLoop:\t Pos 03.2\n");

  // Add one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UB = Builder.CreateAdd(UB, ConstantInt::get(LongType, 1));

  printf("LLVM-IR createParallelLoop:\t Pos 04\n");

  printf("--- AFTER createSubFn!\n");
  LB->dump();
  UB->dump();

  Value *gtid = createCallGlobalThreadNum(dummy_src_loc);
  gtid->dump();
  createCallPushNumThreads(dummy_src_loc, gtid, Builder.getInt32(numThreads));
  //createCallPushNumThreads(dummy_src_loc, gtid, Builder.getInt32(2));
  // Tell the runtime we start a parallel loop
  createCallSpawnThreads(dummy_src_loc, SubFn, LB, UB, Stride, SubFnParam);

  // printf("LLVM-IR createParallelLoop:\t Pos 05\n");

  // TODO: Check if neccessary!
  // Value *myPtr1 = Builder.CreateAlloca(LongType, nullptr, "myPtr1");
  // Value *myPtr2 = Builder.CreateAlloca(LongType, nullptr, "myPtr2");
  // Builder.CreateCall(SubFn, {myPtr1, myPtr2});

  printf("LLVM-IR createParallelLoop:\t Pos 06\n");

  // Mark the end of the lifetime for the parameter struct.
  Type *Ty = Struct->getType();
  ConstantInt *SizeOf = Builder.getInt64(DL.getTypeAllocSize(Ty));
  Builder.CreateLifetimeEnd(Struct, SizeOf);

  printf("LLVM-IR createParallelLoop:\t Pos 07\n");

  freopen("/home/mhalk/ba/dump/moddump_endOfcreateParLoop.ll", "w", stderr);
  M->dump();
  freopen("/dev/tty", "w", stderr);

  printf("LLVM-IR createParallelLoop:\t EXIT\n");

  return IV;
}

void ParallelLoopGeneratorLLVM::createCallSpawnThreads(Value *_loc,
                                                       Value *_microtask,
                                                       Value *LB,
                                                       Value *UB,
                                                       Value *Stride,
                                                       Value *SubFnParam) {
  printf("LLVM-IR createCallSpawnThreads:\tEntry\n");

  // const std::string Name = "GOMP_parallel_loop_runtime_start";
  const std::string Name = "__kmpc_fork_call";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    StructType *identTy = M->getTypeByName("ident_t");
    Type *Kmpc_MicroTy = M->getTypeByName("kmpc_micro");

    if (!Kmpc_MicroTy) {
       // void (*kmpc_micro)(kmp_int32 *global_tid, kmp_int32 *bound_tid,...)
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

  Value *Args[] = {_loc, Builder.getInt32(4), _microtask,
                    LB, UB, Stride, SubFnParam};

  printf("LLVM-IR createCallSpawnThreads:\tFunc defined: Pos 03\n");

  Builder.CreateCall(F, Args);

  printf("LLVM-IR createCallSpawnThreads:\tExit\n");
}

void ParallelLoopGeneratorLLVM::createCallGetWorkItem(Value *loc,
                                                   Value *global_tid,
                                                   Value *pIsLast, Value *pLB,
                                                   Value *pUB, Value *pStride) {
  printf("LLVM-IR createCallGetWorkItem:\tEntry\n");

  const std::string Name = "__kmpc_for_static_init_8";
  Function *F = M->getFunction(Name);
  StructType *identTy = M->getTypeByName("ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {identTy->getPointerTo(), Builder.getInt32Ty(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty()->getPointerTo(),
                      Builder.getInt64Ty()->getPointerTo(),
                      Builder.getInt64Ty()->getPointerTo(),
                      Builder.getInt64Ty()->getPointerTo(),
                      Builder.getInt64Ty(), Builder.getInt64Ty()};

    /*
    Type *Params[] = {identTy->getPointerTo(), Builder.getInt32Ty(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty()->getPointerTo(),
                      Builder.getInt32Ty()->getPointerTo(),
                      Builder.getInt32Ty()->getPointerTo(),
                      Builder.getInt32Ty()->getPointerTo(),
                      Builder.getInt32Ty(), Builder.getInt32Ty()};
                      */

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  printf("LLVM-IR createCallGetWorkItem:\tFunc defined: Pos 05\n");

  // Value *NumberOfThreads = Builder.getInt32(PollyNumThreads);
  Value *Args[] = { loc, global_tid, Builder.getInt32(34), /* Static schedule */
                    pIsLast, pLB, pUB, pStride, Builder.getInt64(1),
                    Builder.getInt64(1) };

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

void ParallelLoopGeneratorLLVM::createCallPushNumThreads(Value *_loc, Value *global_tid,
                              Value *num_threads) {
  const std::string Name = "__kmpc_push_num_threads";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    StructType *identTy = M->getTypeByName("ident_t");

    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {identTy->getPointerTo(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty()};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  printf("LLVM-IR createCallPushNumThreads:\tFunc defined: Pos 02\n");

  Value *Args[] = {_loc, global_tid, num_threads};

  printf("LLVM-IR createCallPushNumThreads:\tFunc defined: Pos 03\n");

  Builder.CreateCall(F, Args);
}

Value *ParallelLoopGeneratorLLVM::createCallGlobalThreadNum(Value *_loc) {
   const std::string Name = "__kmpc_global_thread_num";
   Function *F = M->getFunction(Name);

   // If F is not available, declare it.
   if (!F) {
     StructType *identTy = M->getTypeByName("ident_t");

     GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
     Type *Params[] = {identTy->getPointerTo()};

     FunctionType *Ty = FunctionType::get(Builder.getInt32Ty(), Params, false);
     F = Function::Create(Ty, Linkage, Name, M);
   }

   printf("LLVM-IR createCallGlobalThreadID:\tFunc defined: Pos 02\n");

   Value *Args[] = {_loc};

   printf("LLVM-IR createCallGlobalThreadID:\tFunc defined: Pos 03\n");

   Value *retVal = Builder.CreateCall(F, Args);

   return retVal;
}

Value *ParallelLoopGeneratorLLVM::createSubFn(AllocaInst *StructData,
                  SetVector<Value *> Data, ValueMapT &Map,
                  Function **SubFnPtr, Value *Location) {
  BasicBlock *PrevBB, *HeaderBB, *ExitBB, *CheckNextBB, *PreHeaderBB, *AfterBB;
  Value *LBPtr, *UBPtr, *UserContext, *IDPtr, *ID, *IV, *pIsLast, *pStride;
  Value *LB, *UB, *Stride;
  printf("LLVM-IR createSubFn:\tEntry\n");

  Function *SubFn = createSubFnDefinition();
  LLVMContext &Context = SubFn->getContext();
  int align = 8; // ((dyn_cast<ConstantInt>(UB))->getBitWidth() == 32) ? 4 : 8;

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
  pStride = Builder.CreateAlloca(LongType, nullptr, "polly.par.pStride");

  printf("LLVM-IR createSubFn:\t 02.0\n");

  StructType *va_listTy = M->getTypeByName("va_list");
  StructType *va_listTyX64 = M->getTypeByName("__va_listX64");

  if(!va_listTy) {
    Type *loc_members[] = { Builder.getInt8PtrTy() };

    va_listTy = StructType::create(M->getContext(), loc_members, "va_list", false);
  }

  if(!va_listTyX64) {
    Type *loc_members[] = { Builder.getInt32Ty(), Builder.getInt32Ty(),
                            Builder.getInt8PtrTy(), Builder.getInt8PtrTy()};

    va_listTyX64 = StructType::create(M->getContext(), loc_members, "__va_listX64", false);
  }

  //std::vector<Type *> types(1, Builder.getInt8PtrTy());
  Function *vaStart = Intrinsic::getDeclaration(M, Intrinsic::vastart);
  Function *vaEnd = Intrinsic::getDeclaration(M, Intrinsic::vaend);

  Value *dataPtr = Builder.CreateAlloca(va_listTyX64, nullptr, "polly.par.DATA");
  Value *data = Builder.CreateBitCast(dataPtr, Builder.getInt8PtrTy(), "polly.par.DATA.i8");

  Builder.CreateCall(vaStart, data);

  LB = Builder.CreateVAArg(data, Builder.getInt64Ty(), "polly.par.var_arg.LB");
  UB = Builder.CreateVAArg(data, Builder.getInt64Ty(), "polly.par.var_arg.UB");
  Stride = Builder.CreateVAArg(data, Builder.getInt64Ty(), "polly.par.var_arg.Stride");
  UB->dump();

  Value *userContextPtr = Builder.CreateVAArg(data, Builder.getInt8PtrTy());
  userContextPtr->dump();

  UserContext = Builder.CreateBitCast(
      userContextPtr, StructData->getType(), "polly.par.userContext");

  printf("LLVM-IR createSubFn:\t 02.7\n");

  UserContext->dump();

  extractValuesFromStruct(Data, StructData->getAllocatedType(), UserContext,
                          Map);

  printf("LLVM-IR createSubFn:\t 02.8\n");

  //pLB = Builder.CreateBitCast(LBPtr, Builder.getInt32Ty()->getPointerTo(), "polly.par.LB32");
  //pUB = Builder.CreateBitCast(UBPtr, Builder.getInt32Ty()->getPointerTo(), "polly.par.UB32");

  //Builder.CreateAlignedLoad(LBPtr, align, "polly.par.LB");
  //Builder.CreateAlignedLoad(UBPtr, align, "polly.par.UB");
  ID = Builder.CreateAlignedLoad(IDPtr, align, "polly.par.global_tid");

  // *vUB = ConstantInt::get(Builder.getInt32Ty(), 1, true);
  // *vStride = ConstantInt::get(Builder.getInt32Ty(), 42, true);

  printf("LLVM-IR createSubFn:\t 02.2\n");

  Builder.CreateAlignedStore(LB, LBPtr, align);
  Builder.CreateAlignedStore(UB, UBPtr, align);
  Builder.CreateAlignedStore(Builder.getInt32(0), pIsLast, align);
  Builder.CreateAlignedStore(Stride, pStride, align);

  printf("LLVM-IR createSubFn:\t 02.4\n");

  // Subtract one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  Value *UB_adj = Builder.CreateAdd(UB, ConstantInt::get(LongType, -1),
                         "polly.indvar.UBAdjusted");

  // insert __kmpc_for_static_init_4 HERE
  createCallGetWorkItem(Location, ID, pIsLast, LBPtr, UBPtr, pStride);

  Builder.SetInsertPoint(HeaderBB);

  LB = Builder.CreateAlignedLoad(LBPtr, align, "polly.indvar.init");
  UB = Builder.CreateAlignedLoad(UBPtr, align, "polly.indvar.UB");

  printf("LLVM-IR createSubFn:\t 02.5\n");

  Value *selectCond = Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT, UB, UB_adj, "polly.threadUB_slt_adjUB");
  UB = Builder.CreateSelect(selectCond, UB, UB_adj);
  Builder.CreateAlignedStore(UB, UBPtr, align);

  printf("LLVM-IR createSubFn:\t 02.6\n");

  Builder.CreateCall(vaEnd, data);

  Value *hasIteration = Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SLE, LB, UB, "polly.hasIteration");
  Builder.CreateCondBr(hasIteration, PreHeaderBB, ExitBB);

  //Builder.CreateBr(CheckNextBB);

  // Add code to check if another set of iterations will be executed.
  Builder.SetInsertPoint(CheckNextBB);

  // UB = Builder.CreateSub(UB, ConstantInt::get(LongType, 1), "polly.par.UBAdjusted");
  Builder.CreateBr(ExitBB);

  printf("LLVM-IR createSubFn:\t 04\n");

  Builder.SetInsertPoint(PreHeaderBB);

  Builder.CreateBr(CheckNextBB);
  Builder.SetInsertPoint(&*--Builder.GetInsertPoint());

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
