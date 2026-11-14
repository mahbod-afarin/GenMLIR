//===- GenDramModeSelectPass.cpp ---------------------------------------===//
//
// Group 3 pass: Select GenDRAM execution mode.
// GenDRAM supports two high-level execution styles:
//
//  - Broadcast/homogeneous mode: typical for APSP (pivot broadcast + barriers)
//  - Pipeline/heterogeneous mode: typical for genomics (Search -> Compute queues)
//
// This pass sets a module attribute:
//   gendram.mode = "broadcast" | "pipeline"
//
// Selection rule:
//   - If any dp.phase exists => broadcast
//   - Else if functions named "seeding" and "alignment" exist => pipeline
//   - Else default => broadcast
//
//===----------------------------------------------------------------------===//

#include "genmlir/Dialect/DPGrid/DPGridOps.h"
#include "genmlir/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace genmlir {
namespace {

#define GEN_PASS_DEF_GENDRAMMODESELECT
#include "genmlir/Passes/Passes.h.inc"

struct GenDramModeSelectPass
    : public impl::GenDramModeSelectBase<GenDramModeSelectPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // If already set, do not override.
    if (module->hasAttr("gendram.mode"))
      return;

    bool hasPhase = false;
    module.walk([&](genmlir::dp::PhaseOp) { hasPhase = true; });

    bool hasSeeding = false, hasAlignment = false;
    module.walk([&](func::FuncOp fn) {
      if (fn.getName() == "seeding")
        hasSeeding = true;
      if (fn.getName() == "alignment")
        hasAlignment = true;
    });

    StringRef mode = "broadcast";
    if (!hasPhase && hasSeeding && hasAlignment)
      mode = "pipeline";

    module->setAttr("gendram.mode", StringAttr::get(ctx, mode));
  }
};

} // namespace

std::unique_ptr<Pass> createGenDramModeSelect() {
  return std::make_unique<GenDramModeSelectPass>();
}

} // namespace genmlir
