//===- LoopGenerators.h - IR helper to create loops -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create scalar and OpenMP parallel loops
// as LLVM-IR.
//
//===----------------------------------------------------------------------===//
#ifndef POLLY_LOOP_GENERATORS_LLVM_H
#define POLLY_LOOP_GENERATORS_LLVM_H

#include "polly/CodeGen/IRBuilder.h"
#include "polly/Support/ScopHelper.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/ValueMap.h"

namespace llvm {
class Value;
class Pass;
class BasicBlock;
} // namespace llvm

namespace polly {
using namespace llvm;

/// The ParallelLoopGenerator allows to create parallelized loops
///
/// To parallelize a loop, we perform the following steps:
///   o  Generate a subfunction which will hold the loop body.
///   o  Create a struct to hold all outer values needed in the loop body.
///   o  Create calls to a runtime library to achieve the actual parallelism.
///      These calls will spawn and join threads, define how the work (here the
///      iterations) are distributed between them and make sure each has access
///      to the struct holding all needed values.
///
/// At the moment we support only one parallel runtime, OpenMP.
///
/// If we parallelize the outer loop of the following loop nest,
///
///   S0;
///   for (int i = 0; i < N; i++)
///     for (int j = 0; j < M; j++)
///       S1(i, j);
///   S2;
///
/// we will generate the following code (with different runtime function names):
///
///   S0;
///   auto *values = storeValuesIntoStruct();
///   // Execute subfunction with multiple threads
///   spawn_threads(subfunction, values);
///   join_threads();
///   S2;
///
///  // This function is executed in parallel by different threads
///   void subfunction(values) {
///     while (auto *WorkItem = getWorkItem()) {
///       int LB = WorkItem.begin();
///       int UB = WorkItem.end();
///       for (int i = LB; i < UB; i++)
///         for (int j = 0; j < M; j++)
///           S1(i, j);
///     }
///     cleanup_thread();
///   }
class ParallelLoopGeneratorLLVM: public ParallelLoopGenerator { // TODO: Only for testing.

public:
  /// Create a parallel loop generator for the current function.
  ParallelLoopGeneratorLLVM(PollyIRBuilder &Builder, Pass *P, LoopInfo &LI,
                        DominatorTree &DT, const DataLayout &DL)
      : ParallelLoopGenerator(Builder, P, LI, DT, DL),
      LongType(Type::getIntNTy(Builder.getContext(),
        ((DL.getPointerSizeInBits() > 32) ? 32 : DL.getPointerSizeInBits()))) {}

/*
  Value *createLoop(Value *LowerBound, Value *UpperBound, Value *Stride,
                    PollyIRBuilder &Builder, Pass *P, LoopInfo &LI,
                    DominatorTree &DT, BasicBlock *&ExitBlock,
                    ICmpInst::Predicate Predicate,
                    ScopAnnotator *Annotator = NULL, bool Parallel = false,
                    bool UseGuard = true);
                    */
  Value *createParallelLoop(Value *LB, Value *UB, Value *Stride,
                            SetVector<Value *> &Values, ValueMapT &VMap,
                            BasicBlock::iterator *LoopBody);
  void createCallSpawnThreads(Value *srcLocation, Value *microtask,
                              Value *LB, Value *UB, Value *Stride,
                              Value *SubFnParam);
  void createCallGetWorkItem(Value *loc, Value *global_tid,
                             Value *pIsLast, Value *pLB,
                             Value *pUB, Value *pStride);
  void createCallJoinThreads();
  void createCallCleanupThread(Value *srcLocation, Value *global_tid);
  Value *createSubFn(Value *LB, Value *UB, Value *Stride, AllocaInst *Struct,
                     SetVector<Value *> UsedValues, ValueMapT &VMap,
                     Function **SubFn, Value *Location);

  Value *createCallGlobalThreadNum(Value *srcLocation);
  void createCallPushNumThreads(Value *srcLocation, Value *global_tid,
                                Value *numThreads);

  Function *createSubFnDefinition();

protected:
  /// The type of a "long" on this hardware used for backend calls.
  Type *LongType;

};
} // end namespace polly
#endif
