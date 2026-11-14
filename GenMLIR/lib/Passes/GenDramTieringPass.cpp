//===- GenDramTieringPass.cpp ------------------------------------------===//
//
// Group 3 pass: tier placement and interleaving.
//
// Attaches the memory layout decision to memref.global and memref.alloc ops
// as attributes:
//
//   gendram.tier       : i64
//   gendram.channel    : i64
//   gendram.bank_group : i64
//   gendram.interleave : bool (only if set)
//
// Policy:
//   - objects whose name contains "PTR" or "CAL" go to tier 0 (latency-critical
//     seeding tables)
//   - all other objects go to tier 1
//   - objects with at least one million elements are interleaved
//
// A memref.global is named by its symbol name, a memref.alloc by the optional
// attribute "gendram.object_name". An alloc that already carries gendram.tier
// is left unchanged.
//
//===----------------------------------------------------------------------===//

#include "genmlir/Passes/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace genmlir {
namespace {

#define GEN_PASS_DEF_GENDRAMTIERING
#include "genmlir/Passes/Passes.h.inc"

static bool nameHasPtrCal(StringRef name) {
  std::string upper = name.upper();
  return StringRef(upper).contains("PTR") || StringRef(upper).contains("CAL");
}

static std::optional<int64_t> getStaticNumElements(MemRefType ty) {
  if (!ty.hasStaticShape())
    return std::nullopt;
  int64_t n = 1;
  for (int64_t d : ty.getShape())
    n *= d;
  return n;
}

static void setPlacementAttrs(Operation *op, MLIRContext *ctx,
                              int64_t tier, int64_t channel,
                              int64_t bankGroup, bool interleave) {
  op->setAttr("gendram.tier", IntegerAttr::get(IntegerType::get(ctx, 64), tier));
  op->setAttr("gendram.channel", IntegerAttr::get(IntegerType::get(ctx, 64), channel));
  op->setAttr("gendram.bank_group", IntegerAttr::get(IntegerType::get(ctx, 64), bankGroup));
  if (interleave)
    op->setAttr("gendram.interleave", BoolAttr::get(ctx, true));
}

struct GenDramTieringPass
    : public impl::GenDramTieringBase<GenDramTieringPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // ---- memref.global ----
    module.walk([&](memref::GlobalOp g) {
      std::string name = g.getSymName().str();
      int64_t tier = nameHasPtrCal(name) ? 0 : 1;

      bool interleave = false;
      if (auto mty = dyn_cast<MemRefType>(g.getType())) {
        if (auto elems = getStaticNumElements(mty)) {
          interleave = (*elems >= 1'000'000);
        }
      }

      setPlacementAttrs(g.getOperation(), ctx, tier, /*channel=*/0, /*bankGroup=*/0, interleave);
    });

    // ---- memref.alloc ----
    module.walk([&](memref::AllocOp a) {
      // If already placed, keep it.
      if (a->hasAttr("gendram.tier"))
        return;

      std::string name = "";
      if (auto attr = a->getAttrOfType<StringAttr>("gendram.object_name"))
        name = attr.getValue().str();

      int64_t tier = (!name.empty() && nameHasPtrCal(name)) ? 0 : 1;

      bool interleave = false;
      if (auto mty = dyn_cast<MemRefType>(a.getType())) {
        if (auto elems = getStaticNumElements(mty)) {
          interleave = (*elems >= 1'000'000);
        }
      }
      setPlacementAttrs(a.getOperation(), ctx, tier, /*channel=*/0, /*bankGroup=*/0, interleave);
    });
  }
};

} // namespace

std::unique_ptr<Pass> createGenDramTiering() {
  return std::make_unique<GenDramTieringPass>();
}

} // namespace genmlir
