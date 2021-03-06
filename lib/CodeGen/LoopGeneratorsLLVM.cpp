//===---- LoopGeneratorsLLVM.cpp -  IR helper to create loops -------------===//
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

using namespace llvm;
using namespace polly;

static cl::opt<int>
    PollyNumThreads("polly-lomp-num-threads",
                    cl::desc("Number of threads to use (0 = auto)"),
                    cl::Hidden, cl::init(0));

static cl::opt<int>
    PollyScheduling("polly-lomp-scheduling",
                    cl::desc("Int representation of the KMPC scheduling"),
                    cl::Hidden, cl::init(34));

static cl::opt<int>
    PollyChunkSize("polly-lomp-chunksize",
                    cl::desc("Chunksize to use by the KMPC runtime calls"),
                    cl::Hidden, cl::init(1));

Value *ParallelLoopGeneratorLLVM::createParallelLoop(
    Value *LB, Value *UB, Value *Stride, SetVector<Value *> &UsedValues,
    ValueMapT &Map, BasicBlock::iterator *LoopBody) {

  Function *SubFn;
  AllocaInst *Struct = storeValuesIntoStruct(UsedValues);
  GlobalValue *loc = createSourceLocation(M);

  // int numThreads = (PollyNumThreads > 0) ? PollyNumThreads : 0;

  // Find out which _init/_next/_fini functions to use
  switch (PollyScheduling) {
  	default:
  		PollyScheduling = 34;
  	case 33: // kmp_sch_static_chunked
  	case 34: // kmp_sch_static
  	case 40: // kmp_sch_static_greedy
  	case 41: // kmp_sch_static_balanced
  	case 44: // kmp_sch_static_steal
  	case 45: // kmp_sch_static_balanced_chunked
  		ScheduleType = 0;
  		break;
  	case 35: // kmp_sch_dynamic_chunked
  	case 36: // kmp_sch_guided_chunked
  	case 39: // kmp_sch_trapezoidal
  	case 42: // kmp_sch_guided_iterative_chunked
  	case 43: // kmp_sch_guided_analytical_chunked
  		ScheduleType = 1;
  		break;
  }

  if (LongType->getIntegerBitWidth() != 64) {
    // Truncate the given 64bit integers, when LongType is smaller
    LB = Builder.CreateTrunc(LB, LongType, "polly.truncLB");
    UB = Builder.CreateTrunc(UB, LongType, "polly.truncUB");
    Stride = Builder.CreateTrunc(Stride, LongType, "polly.truncStride");
  }

  Value *SubFnParam = Builder.CreateBitCast(Struct, Builder.getInt8PtrTy(),
                                            "polly.par.userContext");

  BasicBlock::iterator BeforeLoop = Builder.GetInsertPoint();
  Value *IV = createSubFn(Struct, UsedValues, Map, &SubFn, loc);
  *LoopBody = Builder.GetInsertPoint();
  Builder.SetInsertPoint(&*BeforeLoop);

  // Add one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UB = Builder.CreateAdd(UB, ConstantInt::get(LongType, 1));

  // Inform OpenMP runtime about the number of threads
  if (PollyNumThreads > 0) {
    Value *gtid = createCallGlobalThreadNum(loc);
    createCallPushNumThreads(loc, gtid, Builder.getInt32(PollyNumThreads));
  }

  // Tell the runtime we start a parallel loop
  createCallSpawnThreads(loc, SubFn, LB, UB, Stride, SubFnParam);

  // Mark the end of the lifetime for the parameter struct.
  Type *Ty = Struct->getType();
  ConstantInt *SizeOf = Builder.getInt64(DL.getTypeAllocSize(Ty));
  Builder.CreateLifetimeEnd(Struct, SizeOf);

  return IV;
}

void ParallelLoopGeneratorLLVM::createCallSpawnThreads(Value *loc,
                                                       Value *microtask,
                                                       Value *LB,
                                                       Value *UB,
                                                       Value *Stride,
                                                       Value *SubFnParam) {
  const std::string Name = "__kmpc_fork_call";
  Function *F = M->getFunction(Name);
  Type *Kmpc_MicroTy = M->getTypeByName("kmpc_micro");

  if (!Kmpc_MicroTy) {
     // void (*kmpc_micro)(kmp_int32 *global_tid, kmp_int32 *bound_tid,...)
     Type *MicroParams[] = {Builder.getInt32Ty()->getPointerTo(),
                            Builder.getInt32Ty()->getPointerTo()};

     Kmpc_MicroTy = FunctionType::get(Builder.getVoidTy(), MicroParams, true);
  }

  // If F is not available, declare it.
  if (!F) {
    StructType *identTy = M->getTypeByName("struct.ident_t");

    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {identTy->getPointerTo(),
                      Builder.getInt32Ty(),
                      Kmpc_MicroTy->getPointerTo()};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, true);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *task = Builder.CreatePointerBitCastOrAddrSpaceCast(microtask,
    Kmpc_MicroTy->getPointerTo());

  Value *Args[] = {loc, Builder.getInt32(4), task,
                    LB, UB, Stride, SubFnParam};

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorLLVM::createCallStaticInit(Value *loc,
                                                   Value *global_tid,
                                                   Value *pIsLast, Value *pLB,
                                                   Value *pUB, Value *pStride) {

  bool is64bitArch = (LongType->getIntegerBitWidth() == 64);
  const std::string Name = is64bitArch ? "__kmpc_for_static_init_8" :
                                         "__kmpc_for_static_init_4";
  Function *F = M->getFunction(Name);
  StructType *identTy = M->getTypeByName("struct.ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {identTy->getPointerTo(), Builder.getInt32Ty(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty()->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType, LongType};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  // Static schedule
  Value *schedule = Builder.getInt32(34);

  Value *Args[] = {loc, global_tid, schedule, pIsLast, pLB, pUB, pStride,
                   ConstantInt::get(LongType, 1), ConstantInt::get(LongType, 1)};

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorLLVM::createCallStaticFini(Value *loc, Value *id) {
  const std::string Name = "__kmpc_for_static_fini";
  Function *F = M->getFunction(Name);
  StructType *identTy = M->getTypeByName("struct.ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {identTy->getPointerTo(), Builder.getInt32Ty()};
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {loc, id};

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorLLVM::createCallPushNumThreads(Value *loc, Value *id,
                              Value *num_threads) {
  const std::string Name = "__kmpc_push_num_threads";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    StructType *identTy = M->getTypeByName("struct.ident_t");

    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {identTy->getPointerTo(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty()};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {loc, id, num_threads};

  Builder.CreateCall(F, Args);
}

Value *ParallelLoopGeneratorLLVM::createCallGlobalThreadNum(Value *loc) {
   const std::string Name = "__kmpc_global_thread_num";
   Function *F = M->getFunction(Name);

   // If F is not available, declare it.
   if (!F) {
     StructType *identTy = M->getTypeByName("struct.ident_t");

     GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
     Type *Params[] = {identTy->getPointerTo()};

     FunctionType *Ty = FunctionType::get(Builder.getInt32Ty(), Params, false);
     F = Function::Create(Ty, Linkage, Name, M);
   }

   Value *Args[] = {loc};
   Value *retVal = Builder.CreateCall(F, Args);

   return retVal;
}

Value *ParallelLoopGeneratorLLVM::createSubFn(AllocaInst *StructData,
                  SetVector<Value *> Data, ValueMapT &Map,
                  Function **SubFnPtr, Value *Location) {
  BasicBlock *PrevBB, *HeaderBB, *ExitBB, *CheckNextBB, *PreHeaderBB, *AfterBB;
  Value *LBPtr, *UBPtr, *UserContext, *IDPtr, *ID, *IV, *pIsLast, *pStride;
  Value *Sched, *LB, *UB, *Stride, *Shared, *Chunk, *hasWork, *hasIteration;
  Value *adjUB;

  Function *SubFn = createSubFnDefinition();
  LLVMContext &Context = SubFn->getContext();
  bool is64bitArch = (LongType->getIntegerBitWidth() == 64);
  int align = (is64bitArch) ? 8 : 4;
  int chunksize = (PollyChunkSize > 0) ? PollyChunkSize : 1;

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
  pIsLast = Builder.CreateAlloca(Builder.getInt32Ty(), nullptr,
                                 "polly.par.lastIterPtr");
  pStride = Builder.CreateAlloca(LongType, nullptr, "polly.par.StridePtr");

  // Get iterator for retrieving the parameters
  Function::arg_iterator AI = SubFn->arg_begin();
  // First argument holds global thread id. Then move iterator to LB.
  IDPtr = &*AI; std::advance(AI, 2);
  LB = &*AI; std::advance(AI, 1);
  UB = &*AI; std::advance(AI, 1);
  Stride = &*AI; std::advance(AI, 1);
  Shared = &*AI;

  UserContext = Builder.CreateBitCast(
    Shared, StructData->getType(), "polly.par.userContext");

  extractValuesFromStruct(Data, StructData->getAllocatedType(), UserContext,
                          Map);

  ID = Builder.CreateAlignedLoad(IDPtr, align, "polly.par.global_tid");

  Builder.CreateAlignedStore(LB, LBPtr, align);
  Builder.CreateAlignedStore(UB, UBPtr, align);
  Builder.CreateAlignedStore(Builder.getInt32(0), pIsLast, align);
  Builder.CreateAlignedStore(Stride, pStride, align);

  // Subtract one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  adjUB = Builder.CreateAdd(UB, ConstantInt::get(LongType, -1),
                                    "polly.indvar.UBAdjusted");

  Sched = Builder.getInt32(PollyScheduling);
  Chunk = ConstantInt::get(LongType, chunksize);

  if (ScheduleType) {
    // "DYNAMIC" scheduling types are handled below
    UB = adjUB;
    createCallDispatchInit(Location, ID, Sched, LB, UB, Stride, Chunk);
    hasWork = createCallDispatchNext(Location, ID, pIsLast, LBPtr, UBPtr, pStride);
    hasIteration = Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_EQ,
                          hasWork, Builder.getInt32(1), "polly.hasIteration");
    Builder.CreateCondBr(hasIteration, PreHeaderBB, ExitBB);

    Builder.SetInsertPoint(CheckNextBB);
    hasWork = createCallDispatchNext(Location, ID, pIsLast, LBPtr, UBPtr, pStride);
    hasIteration = Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_EQ,
                          hasWork, Builder.getInt32(1), "polly.hasWork");
    Builder.CreateCondBr(hasIteration, PreHeaderBB, ExitBB);

    Builder.SetInsertPoint(PreHeaderBB);
    LB = Builder.CreateAlignedLoad(LBPtr, align, "polly.indvar.init");
    UB = Builder.CreateAlignedLoad(UBPtr, align, "polly.indvar.UB");
  } else {
    // "STATIC" scheduling types are handled below
    createCallStaticInit(Location, ID, pIsLast, LBPtr, UBPtr, pStride);

    LB = Builder.CreateAlignedLoad(LBPtr, align, "polly.indvar.init");
    UB = Builder.CreateAlignedLoad(UBPtr, align, "polly.indvar.UB");

    hasWork = Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT,
                                  UB, adjUB, "polly.UB_slt_adjUB");

    UB = Builder.CreateSelect(hasWork, UB, adjUB);
    Builder.CreateAlignedStore(UB, UBPtr, align);

    hasIteration = Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SLE,
                                             LB, UB, "polly.hasIteration");
    Builder.CreateCondBr(hasIteration, PreHeaderBB, ExitBB);

    Builder.SetInsertPoint(CheckNextBB);
    Builder.CreateBr(ExitBB);

    Builder.SetInsertPoint(PreHeaderBB);
  }

  Builder.CreateBr(CheckNextBB);
  Builder.SetInsertPoint(&*--Builder.GetInsertPoint());
  IV = createLoop(LB, UB, Stride, Builder, P, LI, DT, AfterBB,
                  ICmpInst::ICMP_SLE, nullptr, true, /* UseGuard */ false);

  BasicBlock::iterator LoopBody = Builder.GetInsertPoint();

  // Add code to terminate this subfunction.
  Builder.SetInsertPoint(ExitBB);
  if (ScheduleType == 0) { createCallStaticFini(Location, ID); }
  Builder.CreateRetVoid();

  Builder.SetInsertPoint(&*LoopBody);
  *SubFnPtr = SubFn;

  return IV;
}

Function *ParallelLoopGeneratorLLVM::createSubFnDefinition() {
  Function *F = Builder.GetInsertBlock()->getParent();
  Type *Params[] = {Builder.getInt32Ty()->getPointerTo(),
                         Builder.getInt32Ty()->getPointerTo(),
                         LongType, LongType, LongType, Builder.getInt8PtrTy()};
  FunctionType *FT = FunctionType::get(Builder.getVoidTy(), Params, false);
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
  AI->setName("polly.kmpc.global_tid"); std::advance(AI, 1);
  AI->setName("polly.kmpc.bound_tid"); std::advance(AI, 1);
  AI->setName("polly.kmpc.lb"); std::advance(AI, 1);
  AI->setName("polly.kmpc.ub"); std::advance(AI, 1);
  AI->setName("polly.kmpc.inc"); std::advance(AI, 1);
  AI->setName("polly.kmpc.shared");

  return SubFn;
}

// FIXME: This function only creates a location dummy.
GlobalVariable *ParallelLoopGeneratorLLVM::createSourceLocation(Module *M) {
  const std::string Name = ".loc.dummy";
  GlobalVariable *dummy_src_loc = M->getGlobalVariable(Name);

  if (dummy_src_loc == nullptr) {
    StructType *identTy = M->getTypeByName("struct.ident_t");

    // If the ident_t StructType is not available, declare it.
    // in LLVM-IR: ident_t = type { i32, i32, i32, i32, i8* }
    if (!identTy) {
      Type *loc_members[] = {Builder.getInt32Ty(), Builder.getInt32Ty(),
                             Builder.getInt32Ty(), Builder.getInt32Ty(),
                             Builder.getInt8PtrTy() };

      identTy = StructType::create(M->getContext(), loc_members,
                                   "struct.ident_t", false);
    }

    int strLen = 23;
    auto arrayType = llvm::ArrayType::get(Builder.getInt8Ty(), strLen);

    // Global Variable Definitions
    GlobalVariable* theString = new GlobalVariable(*M, arrayType, true,
                            GlobalValue::PrivateLinkage, 0, ".str.ident");
    theString->setAlignment(1);

    dummy_src_loc = new GlobalVariable(*M, identTy, true,
    GlobalValue::PrivateLinkage, nullptr, ".loc.dummy");
    dummy_src_loc->setAlignment(8);

    // Constant Definitions
    Constant *locInit_str = ConstantDataArray::getString(M->getContext(),
    "Source location dummy.", true);

    Value *stringGEP = Builder.CreateInBoundsGEP(arrayType, theString,
      {Builder.getInt32(0), Builder.getInt32(0)});

    Constant *locInit_struct = ConstantStruct::get(identTy,
      { Builder.getInt32(0), Builder.getInt32(0), Builder.getInt32(0),
        Builder.getInt32(0), (Constant*) stringGEP});

    // Initialize variables
    theString->setInitializer(locInit_str);
    dummy_src_loc->setInitializer(locInit_struct);
  }

  return dummy_src_loc;
}

void ParallelLoopGeneratorLLVM::createCallDispatchInit(Value *loc,
                                                   Value *global_tid,
                                                   Value *Sched, Value *LB,
                                                   Value *UB, Value *Inc,
                                                   Value *Chunk) {

  bool is64bitArch = (LongType->getIntegerBitWidth() == 64);
  const std::string Name = is64bitArch ? "__kmpc_dispatch_init_8" :
                                         "__kmpc_dispatch_init_4";
  Function *F = M->getFunction(Name);
  StructType *identTy = M->getTypeByName("struct.ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {identTy->getPointerTo(), Builder.getInt32Ty(),
                      Builder.getInt32Ty(), LongType, LongType,
                      LongType, LongType};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {loc, global_tid, Sched, LB, UB, Inc, Chunk};

  Builder.CreateCall(F, Args);
}

Value *ParallelLoopGeneratorLLVM::createCallDispatchNext(Value *loc,
                                                   Value *global_tid,
                                                   Value *pIsLast, Value *pLB,
                                                   Value *pUB, Value *pStride) {

  bool is64bitArch = (LongType->getIntegerBitWidth() == 64);
  const std::string Name = is64bitArch ? "__kmpc_dispatch_next_8" :
                                         "__kmpc_dispatch_next_4";
  Function *F = M->getFunction(Name);
  StructType *identTy = M->getTypeByName("struct.ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {identTy->getPointerTo(), Builder.getInt32Ty(),
                      Builder.getInt32Ty()->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo()};

    FunctionType *Ty = FunctionType::get(Builder.getInt32Ty(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {loc, global_tid, pIsLast, pLB, pUB, pStride};

  Value *retVal = Builder.CreateCall(F, Args);
  return retVal;
}
