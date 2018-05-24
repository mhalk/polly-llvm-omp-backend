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
  AllocaInst *Struct = storeValuesIntoStruct(UsedValues);
  GlobalValue *loc = createSourceLocation(M);

  int numThreads = (PollyNumThreads > 0) ? PollyNumThreads : 4;
  numThreads = 4;

  if (LongType->getIntegerBitWidth() != 64) {
    // Truncate the given 64bit integers, when LongType is smaller
    LB = Builder.CreateTrunc(LB, LongType, "polly.truncLB");
    UB = Builder.CreateTrunc(UB, LongType, "polly.truncUB");
    Stride = Builder.CreateTrunc(Stride, LongType, "polly.truncStride");
  }

  BasicBlock::iterator BeforeLoop = Builder.GetInsertPoint();
  Value *IV = createSubFn(Struct, UsedValues, Map, &SubFn, loc);
  *LoopBody = Builder.GetInsertPoint();
  Builder.SetInsertPoint(&*BeforeLoop);

  Value *SubFnParam = Builder.CreateBitCast(Struct, Builder.getInt8PtrTy(),
                                            "polly.par.userContext");

  // Add one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UB = Builder.CreateAdd(UB, ConstantInt::get(LongType, 1));

  // Inform OpenMP runtime about the number of threads
  Value *gtid = createCallGlobalThreadNum(loc);
  createCallPushNumThreads(loc, gtid, Builder.getInt32(numThreads));

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

  // If F is not available, declare it.
  if (!F) {
    StructType *identTy = M->getTypeByName("struct.ident_t");
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

  Value *Args[] = {loc, Builder.getInt32(4), microtask,
                    LB, UB, Stride, SubFnParam};

  Builder.CreateCall(F, Args);

}

void ParallelLoopGeneratorLLVM::createCallGetWorkItem(Value *loc,
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

  Value *Args[] = {loc, global_tid, Builder.getInt32(34), /* Static schedule */
                   pIsLast, pLB, pUB, pStride, ConstantInt::get(LongType, 1),
                   ConstantInt::get(LongType, 1) };

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorLLVM::createCallCleanupThread(Value *loc, Value *id) {
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
  Value *LB, *UB, *Stride;

  Function *SubFn = createSubFnDefinition();
  LLVMContext &Context = SubFn->getContext();
  bool is64bitArch = (LongType->getIntegerBitWidth() == 64);
  int align = (is64bitArch) ? 8 : 4;

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

  // First argument holds global thread id
  IDPtr = &*SubFn->arg_begin();

  LBPtr = Builder.CreateAlloca(LongType, nullptr, "polly.par.LBPtr");
  UBPtr = Builder.CreateAlloca(LongType, nullptr, "polly.par.UBPtr");
  pIsLast = Builder.CreateAlloca(Builder.getInt32Ty(), nullptr,
                                 "polly.par.lastIterPtr");
  pStride = Builder.CreateAlloca(LongType, nullptr, "polly.par.StridePtr");


  StructType *va_listTy = M->getTypeByName("struct.__va_list");

  // If the va_list StructType is not available, declare it.
  // On x64 architecture: va_list = type { i32, i32, i8*, i8* }
  // otherwise: va_list = type { i8* }
  if(!va_listTy) {
    std::vector<Type *> va_list_members;

    if (is64bitArch) {
      va_list_members.push_back(Builder.getInt32Ty());
      va_list_members.push_back(Builder.getInt32Ty());
      va_list_members.push_back(Builder.getInt8PtrTy());
      va_list_members.push_back(Builder.getInt8PtrTy());
    } else {
      va_list_members.push_back(Builder.getInt8PtrTy());
    }

    va_listTy = StructType::create(M->getContext(), va_list_members,
                                   "struct.__va_list", false);
  }

  Function *vaStart = Intrinsic::getDeclaration(M, Intrinsic::vastart);
  Function *vaEnd = Intrinsic::getDeclaration(M, Intrinsic::vaend);

  Value *dataPtr = Builder.CreateAlloca(va_listTy, nullptr, "polly.par.DATA");
  Value *data = Builder.CreateBitCast(dataPtr, Builder.getInt8PtrTy(),
                                      "polly.par.DATA.i8");

  // Load the variable arguments of __kmpc_fork_call
  Builder.CreateCall(vaStart, data);

  LB = Builder.CreateVAArg(data, LongType, "polly.par.var_arg.LB");
  UB = Builder.CreateVAArg(data, LongType, "polly.par.var_arg.UB");
  Stride = Builder.CreateVAArg(data, LongType, "polly.par.var_arg.Stride");
  Value *userContextPtr = Builder.CreateVAArg(data, Builder.getInt8PtrTy());

  Builder.CreateCall(vaEnd, data);

  UserContext = Builder.CreateBitCast(
      userContextPtr, StructData->getType(), "polly.par.userContext");

  extractValuesFromStruct(Data, StructData->getAllocatedType(), UserContext,
                          Map);

  ID = Builder.CreateAlignedLoad(IDPtr, align, "polly.par.global_tid");

  Builder.CreateAlignedStore(LB, LBPtr, align);
  Builder.CreateAlignedStore(UB, UBPtr, align);
  Builder.CreateAlignedStore(Builder.getInt32(0), pIsLast, align);
  Builder.CreateAlignedStore(Stride, pStride, align);

  // Subtract one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  Value *UB_adj = Builder.CreateAdd(UB, ConstantInt::get(LongType, -1),
                                    "polly.indvar.UBAdjusted");

  // Start __kmpc_for_static_init to get the thread-specific params (LB and UB)
  createCallGetWorkItem(Location, ID, pIsLast, LBPtr, UBPtr, pStride);

  Builder.SetInsertPoint(HeaderBB);

  LB = Builder.CreateAlignedLoad(LBPtr, align, "polly.indvar.init");
  UB = Builder.CreateAlignedLoad(UBPtr, align, "polly.indvar.UB");

  Value *selectCond = Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT,
                                         UB, UB_adj, "polly.UB_slt_adjUB");
  UB = Builder.CreateSelect(selectCond, UB, UB_adj);
  Builder.CreateAlignedStore(UB, UBPtr, align);

  Value *hasIteration = Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SLE,
                                           LB, UB, "polly.hasIteration");
  Builder.CreateCondBr(hasIteration, PreHeaderBB, ExitBB);

  // FIXME : CheckNextBB is theoretically not needed anymore.
  // However, it will be removed by other passes and is needed to
  // work with the existing functions.
  Builder.SetInsertPoint(CheckNextBB);
  Builder.CreateBr(ExitBB);

  Builder.SetInsertPoint(PreHeaderBB);
  Builder.CreateBr(CheckNextBB);

  Builder.SetInsertPoint(&*--Builder.GetInsertPoint());

  IV = createLoop(LB, UB, Stride, Builder, P, LI, DT, AfterBB,
                  ICmpInst::ICMP_SLE, nullptr, true, false);
  BasicBlock::iterator LoopBody = Builder.GetInsertPoint();

  // Add code to terminate this subfunction.
  Builder.SetInsertPoint(ExitBB);
  createCallCleanupThread(Location, ID);
  Builder.CreateRetVoid();

  Builder.SetInsertPoint(&*LoopBody);
  *SubFnPtr = SubFn;

  return IV;
}

Function *ParallelLoopGeneratorLLVM::createSubFnDefinition() {
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

  return SubFn;
}

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
