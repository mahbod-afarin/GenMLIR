//===- GenDramMappingPass.cpp ------------------------------------------===//
//
// Group 3 pass: Tile -> PU mapping.
// The paper uses a modulo-style mapping:
//   owner_pu = (ti * M + tj) mod (C * G)
//
// Where:
//   - (ti, tj) are the tile coordinates
//   - C is #channels
//   - G is #bank-groups per channel
//   - M is a logical tiles-per-row factor
//
// This pass annotates each dp.tile with owner_pu.
//
//===----------------------------------------------------------------------===//

#include "genmlir/Dialect/DPGrid/DPGridOps.h"
#include "genmlir/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace genmlir {
namespace {

#define GEN_PASS_DEF_GENDRAMMAPPING
#include "genmlir/Passes/Passes.h.inc"

struct GenDramMappingPass
    : public impl::GenDramMappingBase<GenDramMappingPass> {
  GenDramMappingPass() = default;
  GenDramMappingPass(const GenDramMappingOptions &options)
      : impl::GenDramMappingBase<GenDramMappingPass>(options) {}

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    int64_t denom = static_cast<int64_t>(C) * static_cast<int64_t>(G);
    if (denom <= 0) {
      func.emitError() << "Invalid mapping parameters: C*G must be > 0";
      signalPassFailure();
      return;
    }

    func.walk([&](genmlir::dp::TileOp tile) {
      int64_t ti = tile.getTi();
      int64_t tj = tile.getTj();
      int64_t owner = (ti * static_cast<int64_t>(M) + tj) % denom;

      tile.setOwnerPuAttr(IntegerAttr::get(IntegerType::get(ctx, 64), owner));
    });
  }
};

} // namespace

std::unique_ptr<Pass> createGenDramMapping() {
  return std::make_unique<GenDramMappingPass>();
}

std::unique_ptr<Pass> createGenDramMapping(const GenDramMappingOptions &options) {
  return std::make_unique<GenDramMappingPass>(options);
}

} // namespace genmlir
