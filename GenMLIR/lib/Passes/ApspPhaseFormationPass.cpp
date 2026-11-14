//===- ApspPhaseFormationPass.cpp --------------------------------------===//
//
// Group 2 pass: APSP phase formation.
//
// Blocked Floyd-Warshall runs in three phases per block index (pivot, rowcol,
// internal). The GenDSL translator emits them as dp.phase ops, and this pass
// leaves such functions unchanged. A function without any dp.phase gets its
// whole body wrapped in a single dp.phase(kind="single"), so that
// communication insertion can place a barrier after it. The pass does not
// derive phases from loop structure.
//
//===----------------------------------------------------------------------===//

#include "genmlir/Dialect/DPGrid/DPGridOps.h"
#include "genmlir/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace genmlir {
namespace {

#define GEN_PASS_DEF_APSPPHASEFORMATION
#include "genmlir/Passes/Passes.h.inc"

struct ApspPhaseFormationPass
    : public impl::ApspPhaseFormationBase<ApspPhaseFormationPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    bool hasPhase = false;
    func.walk([&](genmlir::dp::PhaseOp) { hasPhase = true; });
    if (hasPhase)
      return;

    Block &entry = func.getBody().front();
    Operation *terminator = entry.getTerminator();

    llvm::SmallVector<Operation *> toMove;
    for (Operation &op : entry.without_terminator())
      toMove.push_back(&op);

    OpBuilder builder(func.getContext());
    builder.setInsertionPoint(terminator);

    auto phase = builder.create<genmlir::dp::PhaseOp>(
        func.getLoc(), builder.getStringAttr("single"));

    Region &r = phase.getBody();
    Block *b = builder.createBlock(&r);

    builder.setInsertionPointToEnd(b);
    for (Operation *op : toMove)
      op->moveBefore(b, b->end());
    builder.create<genmlir::dp::YieldOp>(func.getLoc());
  }
};

} // namespace

std::unique_ptr<Pass> createApspPhaseFormation() {
  return std::make_unique<ApspPhaseFormationPass>();
}

} // namespace genmlir
