//===- GenDramCommInsertPass.cpp ---------------------------------------===//
//
// Group 4 pass: Insert explicit communication + synchronization.
//
// This pass inserts gendram dialect ops based on the selected module mode.
//
//  - broadcast mode (APSP):
//      * After each dp.phase(kind="pivot"):   broadcast the pivot tile (k,k)
//      * After each dp.phase(kind="rowcol"):  broadcast each updated row/col tile
//      * After every dp.phase:                 insert gendram.barrier(scope="phase")
//
//  - pipeline mode (genomics):
//      * Insert gendram.send(qid=0, payload) near the end of @seeding
//      * Insert gendram.recv(qid=0) at the start of @alignment
//
// Tile payloads:
//
// The payload operand of gendram.broadcast and gendram.send is an SSA value.
// A broadcast takes the global memref (for example @D) as its operand and
// identifies the tile through attributes:
//
//   tile_ti, tile_tj, tile_tk : i64    logical tile coordinates
//   object                    : string name of the global, e.g. "D"
//   payload_kind              : string "tile"
//   phase_kind                : string "pivot" or "rowcol"
//
// The artifact emitter and the simulator read these attributes and treat the
// op as the broadcast of one tile, not of the whole buffer.
//
//===----------------------------------------------------------------------===//

#include "genmlir/Dialect/DPGrid/DPGridOps.h"
#include "genmlir/Dialect/GenDRAM/GenDRAMOps.h"
#include "genmlir/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace genmlir {
namespace {

#define GEN_PASS_DEF_GENDRAMCOMMINSERT
#include "genmlir/Passes/Passes.h.inc"

// Pick the global used as payload of broadcast and send ops: @D or
// @candidates if present, otherwise the first global of the module.
static std::optional<std::pair<StringRef, MemRefType>>
choosePayloadGlobal(ModuleOp module) {
  memref::GlobalOp first;
  memref::GlobalOp preferred;

  module.walk([&](memref::GlobalOp g) {
    if (!first)
      first = g;

    StringRef name = g.getSymName();
    if (name == "D" || name == "candidates")
      preferred = g;
  });

  memref::GlobalOp chosen = preferred ? preferred : first;
  if (!chosen)
    return std::nullopt;

  auto mty = dyn_cast<MemRefType>(chosen.getType());
  if (!mty)
    return std::nullopt;

  return std::make_pair(chosen.getSymName(), mty);
}

static void attachTileAttrs(Operation *op, OpBuilder &builder,
                            StringRef objectName,
                            int64_t ti, int64_t tj,
                            std::optional<uint64_t> tkOpt,
                            StringRef phaseKind) {
  op->setAttr("object", builder.getStringAttr(objectName));
  op->setAttr("payload_kind", builder.getStringAttr("tile"));
  op->setAttr("phase_kind", builder.getStringAttr(phaseKind));
  op->setAttr("tile_ti", builder.getI64IntegerAttr(ti));
  op->setAttr("tile_tj", builder.getI64IntegerAttr(tj));
  if (tkOpt.has_value())
    op->setAttr("tile_tk", builder.getI64IntegerAttr(static_cast<int64_t>(*tkOpt)));
}

static void insertBroadcastAndBarrier(func::FuncOp fn, StringRef payloadName,
                                      MemRefType payloadTy) {
  MLIRContext *ctx = fn.getContext();
  OpBuilder builder(ctx);

  // Walk dp.phase ops in the function, and insert communication right after them.
  fn.walk([&](genmlir::dp::PhaseOp phase) {
    builder.setInsertionPointAfter(phase);

    // Only broadcast when the chosen payload is the distance matrix ("D").
    // (We still insert barriers regardless.)
    bool isDistanceMatrix = (payloadName == "D");

    // Insert broadcasts first (so the barrier is a "phase boundary" marker).
    if (isDistanceMatrix) {
      if (phase.getKind() == "pivot") {
        // Expect exactly one tile inside pivot phase.
        genmlir::dp::TileOp pivotTile;
        phase.getBody().walk([&](genmlir::dp::TileOp t) {
          if (!pivotTile)
            pivotTile = t;
        });

        if (pivotTile) {
          auto getG = builder.create<memref::GetGlobalOp>(
              phase.getLoc(), payloadTy,
              FlatSymbolRefAttr::get(ctx, payloadName));

          // Broadcast the pivot tile (k,k).
          auto bc = builder.create<genmlir::gendram::BroadcastOp>(
              phase.getLoc(), getG.getResult(), /*src_pu=*/IntegerAttr());

          // Set src_pu if we have owner_pu.
          if (auto owner = pivotTile.getOwnerPu()) {
            bc.setSrcPuAttr(builder.getI64IntegerAttr(static_cast<int64_t>(*owner)));
          }

          attachTileAttrs(bc.getOperation(), builder, payloadName,
                          pivotTile.getTi(), pivotTile.getTj(),
                          pivotTile.getTk(), "pivot");
        }
      } else if (phase.getKind() == "rowcol") {
        // Broadcast each updated row/col tile after the rowcol phase completes.
        // This models the propagation of row/col blocks used by the internal phase.
        llvm::SmallVector<genmlir::dp::TileOp, 16> tiles;
        phase.getBody().walk([&](genmlir::dp::TileOp t) { tiles.push_back(t); });

        if (!tiles.empty()) {
          auto getG = builder.create<memref::GetGlobalOp>(
              phase.getLoc(), payloadTy,
              FlatSymbolRefAttr::get(ctx, payloadName));

          for (auto t : tiles) {
            auto bc = builder.create<genmlir::gendram::BroadcastOp>(
                phase.getLoc(), getG.getResult(), /*src_pu=*/IntegerAttr());

            if (auto owner = t.getOwnerPu()) {
              bc.setSrcPuAttr(builder.getI64IntegerAttr(static_cast<int64_t>(*owner)));
            }

            attachTileAttrs(bc.getOperation(), builder, payloadName,
                            t.getTi(), t.getTj(), t.getTk(), "rowcol");
          }
        }
      }
    }

    // Barrier after every phase.
    builder.create<genmlir::gendram::BarrierOp>(
        phase.getLoc(), /*scope=*/builder.getStringAttr("phase"));
  });
}

static void insertSendRecv(ModuleOp module, StringRef payloadName,
                           MemRefType payloadTy) {
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);

  func::FuncOp seeding = module.lookupSymbol<func::FuncOp>("seeding");
  func::FuncOp alignment = module.lookupSymbol<func::FuncOp>("alignment");

  if (!seeding || !alignment)
    return;

  // Insert recv at start of alignment (after entry block arguments).
  {
    Block &entry = alignment.getBody().front();
    builder.setInsertionPointToStart(&entry);

    builder.create<genmlir::gendram::RecvOp>(
        alignment.getLoc(), /*qid=*/builder.getI64IntegerAttr(0));
  }

  // Insert send near end of seeding (right before func.return).
  {
    Block &entry = seeding.getBody().front();
    Operation *term = entry.getTerminator();
    builder.setInsertionPoint(term);

    auto getG = builder.create<memref::GetGlobalOp>(
        seeding.getLoc(), payloadTy, FlatSymbolRefAttr::get(ctx, payloadName));

    builder.create<genmlir::gendram::SendOp>(
        seeding.getLoc(), getG.getResult(), /*qid=*/builder.getI64IntegerAttr(0));
  }
}

struct GenDramCommInsertPass
    : public impl::GenDramCommInsertBase<GenDramCommInsertPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    auto modeAttr = module->getAttrOfType<StringAttr>("gendram.mode");
    StringRef mode = modeAttr ? modeAttr.getValue() : "broadcast";

    auto payload = choosePayloadGlobal(module);

    if (mode == "broadcast") {
      if (!payload) {
        // Insert barriers after phases only.
        module.walk([&](func::FuncOp fn) {
          OpBuilder builder(ctx);
          fn.walk([&](genmlir::dp::PhaseOp phase) {
            builder.setInsertionPointAfter(phase);
            builder.create<genmlir::gendram::BarrierOp>(
                phase.getLoc(), builder.getStringAttr("phase"));
          });
        });
        return;
      }

      module.walk([&](func::FuncOp fn) {
        insertBroadcastAndBarrier(fn, payload->first, payload->second);
      });
      return;
    }

    // pipeline mode
    if (mode == "pipeline" && payload) {
      insertSendRecv(module, payload->first, payload->second);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createGenDramCommInsert() {
  return std::make_unique<GenDramCommInsertPass>();
}

} // namespace genmlir
