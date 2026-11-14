//===- DpTilingPass.cpp -------------------------------------------------===//
//
// Group 2 pass: tiling.
//
// The GenDSL translator emits dp.tile ops with explicit (ti, tj) coordinates,
// and this pass leaves such functions unchanged. A function without any
// dp.tile gets its whole body wrapped in a single dp.tile(ti=0, tj=0), so that
// the mapping and schedule passes always find a tile to annotate.
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

#define GEN_PASS_DEF_DPTILING
#include "genmlir/Passes/Passes.h.inc"

struct DpTilingPass : public impl::DpTilingBase<DpTilingPass> {
  DpTilingPass() = default;
  DpTilingPass(const DpTilingOptions &options)
      : impl::DpTilingBase<DpTilingPass>(options) {}

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    bool hasTile = false;
    func.walk([&](genmlir::dp::TileOp) { hasTile = true; });

    if (hasTile)
      return;

    Block &entry = func.getBody().front();
    Operation *terminator = entry.getTerminator();

    // Collect ops to move (everything except the terminator).
    llvm::SmallVector<Operation *> toMove;
    for (Operation &op : entry.without_terminator())
      toMove.push_back(&op);

    OpBuilder builder(func.getContext());
    builder.setInsertionPoint(terminator);

    // Create the dp.tile wrapper right before func.return.
    auto tile = builder.create<genmlir::dp::TileOp>(
        func.getLoc(),
        /*ti=*/builder.getI64IntegerAttr(0),
        /*tj=*/builder.getI64IntegerAttr(0),
        /*tk=*/IntegerAttr(),
        /*owner_pu=*/IntegerAttr());

    // Create the tile body block and move ops into it.
    Region &tileRegion = tile.getBody();
    Block *tileBlock = builder.createBlock(&tileRegion);

    builder.setInsertionPointToEnd(tileBlock);
    for (Operation *op : toMove)
      op->moveBefore(tileBlock, tileBlock->end());

    builder.create<genmlir::dp::YieldOp>(func.getLoc());
  }
};

} // namespace

std::unique_ptr<Pass> createDpTiling() {
  return std::make_unique<DpTilingPass>();
}

std::unique_ptr<Pass> createDpTiling(const DpTilingOptions &options) {
  return std::make_unique<DpTilingPass>(options);
}

} // namespace genmlir
