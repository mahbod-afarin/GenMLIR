//===- DpCanonicalizePass.cpp ------------------------------------------===//
//
// Group 1 pass: DP normalization.
//
// Checks the semiring attribute of every dp.grid_update. A missing or unknown
// value is replaced by min_plus, so later passes can rely on the attribute.
//
//===----------------------------------------------------------------------===//

#include "genmlir/Dialect/DPGrid/DPGridOps.h"
#include "genmlir/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace genmlir {
namespace {

#define GEN_PASS_DEF_DPCANONICALIZE
#include "genmlir/Passes/Passes.h.inc"

static bool isAllowedSemiring(StringRef s) {
  return s == "min_plus" || s == "max_plus";
}

struct DpCanonicalizePass : public impl::DpCanonicalizeBase<DpCanonicalizePass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    func.walk([&](genmlir::dp::GridUpdateOp op) {
      auto sem = op.getSemiringAttr();
      if (!sem || !isAllowedSemiring(sem.getValue())) {
        // Default to min_plus if the attribute is missing or unknown.
        op.setSemiringAttr(StringAttr::get(ctx, "min_plus"));
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createDpCanonicalize() {
  return std::make_unique<DpCanonicalizePass>();
}

} // namespace genmlir
