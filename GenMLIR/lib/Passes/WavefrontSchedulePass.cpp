//===- WavefrontSchedulePass.cpp ---------------------------------------===//
//
// Group 2 pass: wavefront scheduling for alignment.
//
// The dependencies of alignment-style DP require a diagonal (wavefront) order.
// The pass annotates every dp.tile with
//   wavefront_step = ti + tj
// and does not reorder the IR. The artifact emitter writes the step into the
// schedule.
//
//===----------------------------------------------------------------------===//

#include "genmlir/Dialect/DPGrid/DPGridOps.h"
#include "genmlir/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace genmlir {
namespace {

#define GEN_PASS_DEF_WAVEFRONTSCHEDULE
#include "genmlir/Passes/Passes.h.inc"

struct WavefrontSchedulePass
    : public impl::WavefrontScheduleBase<WavefrontSchedulePass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    func.walk([&](genmlir::dp::TileOp tile) {
      int64_t ti = tile.getTi();
      int64_t tj = tile.getTj();
      int64_t step = ti + tj;

      tile->setAttr("wavefront_step", IntegerAttr::get(IntegerType::get(ctx, 64), step));
    });
  }
};

} // namespace

std::unique_ptr<Pass> createWavefrontSchedule() {
  return std::make_unique<WavefrontSchedulePass>();
}

} // namespace genmlir
