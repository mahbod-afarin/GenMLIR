//===- GenDramEmitArtifactsPass.cpp ------------------------------------===//
//
// Back-end pass: emit the compiler artifacts.
//
//   layout.json        memory layout plan
//   pu_programs/*.txt  tiles assigned to each PU
//   schedule.json      compute, communication and synchronization events
//
// The pass reads the IR after mapping, tiering and communication insertion and
// writes the artifacts to the output directory.
//
// schedule.json carries what the GenDRAM simulator needs for timing and for
// functional execution. Every compute event names its tile as a subview of
// the global object (coordinates, offsets, shape), its operand tiles, and the
// number of row-buffer reads and writes.
//
//===----------------------------------------------------------------------===//

#include "genmlir/Dialect/DPGrid/DPGridOps.h"
#include "genmlir/Dialect/GenDRAM/GenDRAMOps.h"
#include "genmlir/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>

using namespace mlir;

namespace genmlir {
namespace {

#define GEN_PASS_DEF_GENDRAMEMITARTIFACTS
#include "genmlir/Passes/Passes.h.inc"

static std::string jsonEscape(StringRef s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
    case '\\': out += "\\\\"; break;
    case '"':  out += "\\\""; break;
    case '\n': out += "\\n"; break;
    case '\r': out += "\\r"; break;
    case '\t': out += "\\t"; break;
    default:   out += c; break;
    }
  }
  return out;
}

static std::optional<int64_t> getI64Attr(Operation *op, StringRef name) {
  if (auto a = op->getAttrOfType<IntegerAttr>(name))
    return a.getInt();
  return std::nullopt;
}

static bool getBoolAttr(Operation *op, StringRef name) {
  if (auto a = op->getAttrOfType<BoolAttr>(name))
    return a.getValue();
  return false;
}

// Print an MLIR type into a stable string.
static std::string typeToString(Type t) {
  std::string s;
  llvm::raw_string_ostream os(s);
  t.print(os);
  os.flush();
  return s;
}

static int64_t elemTypeBytes(Type elemTy) {
  if (auto it = dyn_cast<IntegerType>(elemTy)) {
    return (it.getWidth() + 7) / 8;
  }
  if (auto ft = dyn_cast<FloatType>(elemTy)) {
    return (ft.getWidth() + 7) / 8;
  }
  return 4; // conservative default
}

static int64_t optU64ToI64(std::optional<uint64_t> v) {
  return v.has_value() ? static_cast<int64_t>(*v) : -1;
}

struct Global2DInfo {
  int64_t dim0 = -1;
  int64_t dim1 = -1;
  int64_t elemBytes = 4;
  std::string elemType;
  std::string mlirType;
};

static std::optional<Global2DInfo> findGlobal2D(ModuleOp module, StringRef name) {
  std::optional<Global2DInfo> out;
  module.walk([&](memref::GlobalOp g) {
    if (g.getSymName() != name)
      return;
    auto mty = dyn_cast<MemRefType>(g.getType());
    if (!mty)
      return;
    if (mty.getRank() < 2)
      return;

    Global2DInfo info;
    if (mty.hasStaticShape()) {
      info.dim0 = mty.getShape()[0];
      info.dim1 = mty.getShape()[1];
    }
    info.elemBytes = elemTypeBytes(mty.getElementType());
    info.elemType = typeToString(mty.getElementType());
    info.mlirType = typeToString(mty);
    out = info;
  });
  return out;
}

struct TileView {
  int64_t i0 = -1;
  int64_t j0 = -1;
  int64_t rows = -1;
  int64_t cols = -1;
};

static TileView tileView2D(int64_t ti, int64_t tj, int64_t B,
                           int64_t N0, int64_t N1) {
  TileView v;
  if (B <= 0 || N0 < 0 || N1 < 0)
    return v;

  v.i0 = ti * B;
  v.j0 = tj * B;

  if (v.i0 >= N0 || v.j0 >= N1) {
    v.rows = 0;
    v.cols = 0;
    return v;
  }

  v.rows = std::min<int64_t>(B, N0 - v.i0);
  v.cols = std::min<int64_t>(B, N1 - v.j0);
  return v;
}

static int64_t kBlockLen(int64_t k, int64_t B, int64_t N0) {
  if (k < 0 || B <= 0 || N0 < 0)
    return -1;
  int64_t k0 = k * B;
  if (k0 >= N0)
    return 0;
  return std::min<int64_t>(B, N0 - k0);
}

struct TileInfo {
  int64_t ti = 0;
  int64_t tj = 0;
  int64_t tk = -1;
  int64_t owner = -1;
  int64_t wavefrontStep = -1;
  int64_t numUpdates = 0;
  std::string phaseKind;
};

struct GenDramEmitArtifactsPass
    : public impl::GenDramEmitArtifactsBase<GenDramEmitArtifactsPass> {
  GenDramEmitArtifactsPass() = default;
  GenDramEmitArtifactsPass(const GenDramEmitArtifactsOptions &options)
      : impl::GenDramEmitArtifactsBase<GenDramEmitArtifactsPass>(options) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Create output directories.
    if (std::error_code ec = llvm::sys::fs::create_directories(outDir)) {
      module.emitError() << "Failed to create out_dir: " << outDir
                         << " (" << ec.message() << ")";
      signalPassFailure();
      return;
    }

    std::string puDir = (llvm::Twine(outDir) + "/pu_programs").str();
    llvm::sys::fs::create_directories(puDir);

    emitLayout(module);
    emitPUPrograms(module, puDir);
    emitSchedule(module);
  }

  void emitLayout(ModuleOp module) {
    std::string path = (llvm::Twine(outDir) + "/layout.json").str();
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      module.emitError() << "Failed to open " << path << ": " << ec.message();
      signalPassFailure();
      return;
    }

    os << "{\n";
    os << "  \"objects\": [\n";

    bool first = true;

    // memref.global objects
    module.walk([&](memref::GlobalOp g) {
      if (!first) os << ",\n";
      first = false;

      Operation *op = g.getOperation();
      int64_t tier = getI64Attr(op, "gendram.tier").value_or(-1);
      int64_t ch   = getI64Attr(op, "gendram.channel").value_or(0);
      int64_t bg   = getI64Attr(op, "gendram.bank_group").value_or(0);
      bool interleave = getBoolAttr(op, "gendram.interleave");

      os << "    {";
      os << "\"name\":\"" << jsonEscape(g.getSymName()) << "\",";
      os << "\"kind\":\"global\",";
      os << "\"mlir_type\":\"" << jsonEscape(typeToString(g.getType())) << "\",";
      os << "\"tier\":" << tier << ",";
      os << "\"channel\":" << ch << ",";
      os << "\"bank_group\":" << bg << ",";
      os << "\"interleave\":" << (interleave ? "true" : "false");
      os << "}";
    });

    // memref.alloc objects
    module.walk([&](memref::AllocOp a) {
      // Use the explicit name if present, else a placeholder.
      std::string name = "alloc";
      if (auto n = a->getAttrOfType<StringAttr>("gendram.object_name"))
        name = n.getValue().str();

      Operation *op = a.getOperation();
      int64_t tier = getI64Attr(op, "gendram.tier").value_or(-1);
      int64_t ch   = getI64Attr(op, "gendram.channel").value_or(0);
      int64_t bg   = getI64Attr(op, "gendram.bank_group").value_or(0);
      bool interleave = getBoolAttr(op, "gendram.interleave");

      if (!first) os << ",\n";
      first = false;

      os << "    {";
      os << "\"name\":\"" << jsonEscape(name) << "\",";
      os << "\"kind\":\"alloc\",";
      os << "\"mlir_type\":\"" << jsonEscape(typeToString(a.getType())) << "\",";
      os << "\"tier\":" << tier << ",";
      os << "\"channel\":" << ch << ",";
      os << "\"bank_group\":" << bg << ",";
      os << "\"interleave\":" << (interleave ? "true" : "false");
      os << "}";
    });

    os << "\n  ]\n";
    os << "}\n";
  }

  void emitPUPrograms(ModuleOp module, StringRef puDir) {
    // Group tiles by owner_pu.
    llvm::MapVector<int64_t, std::vector<TileInfo>> byPU;

    module.walk([&](genmlir::dp::TileOp tile) {
      TileInfo info;
      info.ti = tile.getTi();
      info.tj = tile.getTj();
      info.owner = optU64ToI64(tile.getOwnerPu());
      info.tk = tile.getTk().has_value() ? static_cast<int64_t>(*tile.getTk()) : -1;

      if (auto wf = tile->getAttrOfType<IntegerAttr>("wavefront_step"))
        info.wavefrontStep = wf.getInt();

      if (auto phase = tile->getParentOfType<genmlir::dp::PhaseOp>())
        info.phaseKind = phase.getKind().str();

      // Prefer a precomputed update estimate if present.
      if (auto upd = tile->getAttrOfType<IntegerAttr>("gendram.dp_updates")) {
        info.numUpdates = upd.getInt();
      } else {
        // Otherwise count dp.grid_update ops inside tile body.
        int64_t updates = 0;
        tile.getBody().walk([&](genmlir::dp::GridUpdateOp) { updates++; });
        info.numUpdates = updates;
      }

      byPU[info.owner].push_back(info);
    });

    // Write one file per PU id (skip owner=-1).
    for (auto &kv : byPU) {
      int64_t owner = kv.first;
      if (owner < 0)
        continue;

      std::string path =
          (llvm::Twine(puDir) + "/pu_" + llvm::Twine(owner) + ".txt").str();
      std::error_code ec;
      llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        module.emitError() << "Failed to open " << path << ": " << ec.message();
        signalPassFailure();
        return;
      }

      os << "PU Program (reference)\n";
      os << "======================\n";
      os << "owner_pu: " << owner << "\n\n";

      os << "Tiles assigned to this PU:\n";
      for (auto &t : kv.second) {
        os << "  - tile(ti=" << t.ti << ", tj=" << t.tj;
        if (t.tk >= 0)
          os << ", tk=" << t.tk;
        os << ")";
        if (!t.phaseKind.empty())
          os << " phase=" << t.phaseKind;
        if (t.wavefrontStep >= 0)
          os << " wavefront_step=" << t.wavefrontStep;
        os << " dp_updates=" << t.numUpdates << "\n";
      }
    }

    // Index file describing the directory.
    std::string indexPath = (llvm::Twine(puDir) + "/README.txt").str();
    std::error_code ec;
    llvm::raw_fd_ostream idx(indexPath, ec, llvm::sys::fs::OF_Text);
    if (!ec) {
      idx << "This directory contains one text file per PU (pu_<id>.txt).\n";
      idx << "Each file lists the tiles assigned to that PU and a dp_update estimate.\n";
    }
  }

  void emitSchedule(ModuleOp module) {
    std::string path = (llvm::Twine(outDir) + "/schedule.json").str();
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      module.emitError() << "Failed to open " << path << ": " << ec.message();
      signalPassFailure();
      return;
    }

    StringRef mode = "unknown";
    if (auto m = module->getAttrOfType<StringAttr>("gendram.mode"))
      mode = m.getValue();

    StringRef workload = "unknown";
    if (auto w = module->getAttrOfType<StringAttr>("gendram.workload"))
      workload = w.getValue();

    int64_t tileSize = -1;
    if (auto ts = module->getAttrOfType<IntegerAttr>("gendram.tile_size"))
      tileSize = ts.getInt();

    // Workload-specific primary matrix object:
    //   - APSP:      @D (distance matrix)
    //   - Alignment: @H (DP score table)
    auto dInfo = findGlobal2D(module, "D");
    auto hInfo = findGlobal2D(module, "H");

    os << "{\n";
    os << "  \"mode\": \"" << jsonEscape(mode) << "\",\n";
    os << "  \"workload\": \"" << jsonEscape(workload) << "\",\n";
    if (tileSize >= 0)
      os << "  \"tile_size\": " << tileSize << ",\n";
    // Primary matrix metadata.
    if (workload == "apsp") {
      if (dInfo && dInfo->dim0 >= 0 && dInfo->dim1 >= 0) {
        os << "  \"D_shape\": [" << dInfo->dim0 << ", " << dInfo->dim1 << "],\n";
        os << "  \"elem_bytes\": " << dInfo->elemBytes << ",\n";
        os << "  \"elem_type\": \"" << jsonEscape(dInfo->elemType) << "\",\n";
      }
    } else if (workload == "alignment") {
      if (hInfo && hInfo->dim0 >= 0 && hInfo->dim1 >= 0) {
        os << "  \"H_shape\": [" << hInfo->dim0 << ", " << hInfo->dim1 << "],\n";
        os << "  \"elem_bytes\": " << hInfo->elemBytes << ",\n";
        os << "  \"elem_type\": \"" << jsonEscape(hInfo->elemType) << "\",\n";
      }

      // Emit common alignment parameters if present.
      if (auto q = module->getAttrOfType<IntegerAttr>("gendram.query_len"))
        os << "  \"query_len\": " << q.getInt() << ",\n";
      if (auto r = module->getAttrOfType<IntegerAttr>("gendram.ref_len"))
        os << "  \"ref_len\": " << r.getInt() << ",\n";
      if (auto a = module->getAttrOfType<StringAttr>("gendram.algo"))
        os << "  \"algo\": \"" << jsonEscape(a.getValue()) << "\",\n";
      if (auto a = module->getAttrOfType<IntegerAttr>("gendram.match"))
        os << "  \"match\": " << a.getInt() << ",\n";
      if (auto a = module->getAttrOfType<IntegerAttr>("gendram.mismatch"))
        os << "  \"mismatch\": " << a.getInt() << ",\n";
      if (auto a = module->getAttrOfType<IntegerAttr>("gendram.gap"))
        os << "  \"gap\": " << a.getInt() << ",\n";
      if (auto a = module->getAttrOfType<IntegerAttr>("gendram.band_width"))
        os << "  \"band_width\": " << a.getInt() << ",\n";
    }
    os << "  \"events\": [\n";

    bool firstEvent = true;

    auto emitComma = [&]() {
      if (!firstEvent)
        os << ",\n";
      firstEvent = false;
    };

    auto emitTileRef = [&](StringRef object, int64_t ti, int64_t tj, int64_t tk,
                           StringRef source, StringRef bcastPhase) {
      os << "{";
      os << "\"object\":\"" << jsonEscape(object) << "\",";
      os << "\"ti\":" << ti << ",";
      os << "\"tj\":" << tj;
      if (tk >= 0)
        os << ",\"tk\":" << tk;
      os << ",\"source\":\"" << jsonEscape(source) << "\"";
      if (!bcastPhase.empty())
        os << ",\"broadcast_phase\":\"" << jsonEscape(bcastPhase) << "\"";
      os << "}";
    };

    module.walk([&](func::FuncOp fn) {
      fn.walk([&](Operation *op) {
        // ---- Compute events (dp.tile) ----
        if (auto tile = dyn_cast<genmlir::dp::TileOp>(op)) {
          emitComma();

          int64_t owner = optU64ToI64(tile.getOwnerPu());
          int64_t ti = tile.getTi();
          int64_t tj = tile.getTj();
          int64_t tk = tile.getTk().has_value() ? static_cast<int64_t>(*tile.getTk()) : -1;

          std::string phaseKind = "";
          if (auto phase = tile->getParentOfType<genmlir::dp::PhaseOp>())
            phaseKind = phase.getKind().str();

          // Wavefront step (alignment)
          int64_t wfStep = -1;
          if (auto wf = tile->getAttrOfType<IntegerAttr>("wavefront_step"))
            wfStep = wf.getInt();

          // Derive tile view for the primary matrix object.
          // APSP uses D; alignment uses H.
          TileView v;
          int64_t kLen = -1;
          bool havePrimary = false;
          int64_t primaryElemBytes = 4;
          StringRef primaryName = "";
          int64_t primaryDim0 = -1, primaryDim1 = -1;

          if (workload == "apsp" && dInfo && tileSize > 0) {
            primaryName = "D";
            primaryDim0 = dInfo->dim0;
            primaryDim1 = dInfo->dim1;
            primaryElemBytes = dInfo->elemBytes;
            havePrimary = (primaryDim0 >= 0 && primaryDim1 >= 0);
            v = tileView2D(ti, tj, tileSize, primaryDim0, primaryDim1);
            kLen = kBlockLen(tk, tileSize, primaryDim0);
          } else if (workload == "alignment" && hInfo && tileSize > 0) {
            primaryName = "H";
            primaryDim0 = hInfo->dim0;
            primaryDim1 = hInfo->dim1;
            primaryElemBytes = hInfo->elemBytes;
            havePrimary = (primaryDim0 >= 0 && primaryDim1 >= 0);
            v = tileView2D(ti, tj, tileSize, primaryDim0, primaryDim1);
          }

          // dp_updates estimate.
          int64_t updates = 0;
          if (workload == "apsp" && v.rows >= 0 && v.cols >= 0 && kLen >= 0) {
            updates = v.rows * v.cols * kLen;
          } else if (workload == "alignment" && v.rows >= 0 && v.cols >= 0) {
            // Smith-Waterman cell updates per tile, without the boundary
            // row and column of the table.
            int64_t effRows = v.rows - (v.i0 == 0 ? 1 : 0);
            int64_t effCols = v.cols - (v.j0 == 0 ? 1 : 0);
            updates = std::max<int64_t>(0, effRows) * std::max<int64_t>(0, effCols);
          } else {
            // fallback: count dp.grid_update ops in region
            tile.getBody().walk([&](genmlir::dp::GridUpdateOp) { updates++; });
          }

          // The semiring follows from the workload.
          std::string semiring = "";
          if (workload == "apsp")
            semiring = "min_plus";
          else if (workload == "alignment")
            semiring = "max_plus";

          os << "    {";
          os << "\"fn\":\"" << jsonEscape(fn.getName()) << "\",";
          os << "\"op\":\"compute_tile\",";
          os << "\"owner_pu\":" << owner << ",";
          os << "\"ti\":" << ti << ",";
          os << "\"tj\":" << tj;
          if (tk >= 0)
            os << ",\"tk\":" << tk;
          os << ",\"dp_updates\":" << updates;
          if (!semiring.empty())
            os << ",\"semiring\":\"" << jsonEscape(semiring) << "\"";
          if (!phaseKind.empty())
            os << ",\"phase\":\"" << jsonEscape(phaseKind) << "\"";
          if (wfStep >= 0)
            os << ",\"wavefront_step\":" << wfStep;

          // Tile addressing / subview
          if (havePrimary && tileSize > 0 && v.rows >= 0 && v.cols >= 0) {
            os << ",\"tile\":{";
            os << "\"object\":\"" << jsonEscape(primaryName) << "\",";
            os << "\"i0\":" << v.i0 << ",\"j0\":" << v.j0 << ",";
            os << "\"rows\":" << v.rows << ",\"cols\":" << v.cols << ",";
            os << "\"elem_bytes\":" << primaryElemBytes;
            os << "}";
          }

          // Row-buffer ops summary (per updated/output tile).
          if (v.rows >= 0) {
            os << ",\"rowbuf\":{";
            os << "\"read_rows\":" << v.rows << ",";
            os << "\"write_rows\":" << v.rows;
            os << "}";
          }

          // APSP-specific operand metadata so the simulator can run functionally.
          if (workload == "apsp" && tk >= 0) {
            os << ",\"k_len\":" << kLen;
            os << ",\"inputs\":[";

            // Updated tile is always read from DRAM.
            // (Other operand tiles are expected to come from broadcasts.)
            bool firstIn = true;
            auto sep = [&]() {
              if (!firstIn)
                os << ",";
              firstIn = false;
            };

            if (phaseKind == "pivot") {
              // pivot depends only on itself.
              sep();
              emitTileRef("D", ti, tj, tk, "dram", "");
            } else if (phaseKind == "rowcol") {
              // rowcol depends on pivot (k,k) from pivot broadcast.
              // plus the tile itself from DRAM.
              sep();
              emitTileRef("D", tk, tk, tk, "broadcast", "pivot");
              sep();
              emitTileRef("D", ti, tj, tk, "dram", "");
            } else if (phaseKind == "internal") {
              // internal depends on row (k, j) and col (i, k) from rowcol broadcast,
              // plus the tile itself from DRAM.
              sep();
              emitTileRef("D", ti, tk, tk, "broadcast", "rowcol");
              sep();
              emitTileRef("D", tk, tj, tk, "broadcast", "rowcol");
              sep();
              emitTileRef("D", ti, tj, tk, "dram", "");
            } else {
              // Unknown phase: just say it reads itself.
              sep();
              emitTileRef("D", ti, tj, tk, "dram", "");
            }

            os << "]";

            // Output tile
            os << ",\"output\":";
            emitTileRef("D", ti, tj, tk, "dram", "");
          }

          // Alignment-specific operand metadata (neighbor tile deps).
          if (workload == "alignment" && havePrimary && v.rows >= 0 && v.cols >= 0) {
            os << ",\"inputs\":[";
            bool firstIn = true;
            auto sep = [&]() {
              if (!firstIn)
                os << ",";
              firstIn = false;
            };

            // Top tile
            if (ti > 0) {
              sep();
              emitTileRef("H", ti - 1, tj, /*tk=*/-1, "dram", "");
            }
            // Left tile
            if (tj > 0) {
              sep();
              emitTileRef("H", ti, tj - 1, /*tk=*/-1, "dram", "");
            }
            // Diagonal tile
            if (ti > 0 && tj > 0) {
              sep();
              emitTileRef("H", ti - 1, tj - 1, /*tk=*/-1, "dram", "");
            }
            os << "]";

            // Output tile
            os << ",\"output\":";
            emitTileRef("H", ti, tj, /*tk=*/-1, "dram", "");
          }

          os << "}";
          return;
        }

        // Helper for non-tile events
        auto recordEvent = [&](StringRef kind, std::function<void(int64_t &tiOut, int64_t &tjOut, int64_t &tkOut)> overrideCoords,
                               std::function<void()> extraFields) {
          emitComma();

          int64_t owner = -1;
          int64_t ti = -1, tj = -1, tk = -1;

          // If nested under a tile, inherit coords/owner.
          if (auto parentTile = op->getParentOfType<genmlir::dp::TileOp>()) {
            owner = optU64ToI64(parentTile.getOwnerPu());
            ti = parentTile.getTi();
            tj = parentTile.getTj();
            tk = parentTile.getTk().has_value() ? static_cast<int64_t>(*parentTile.getTk()) : -1;
          }

          // Let caller override ti/tj/tk (used for broadcast ops emitted after phases).
          overrideCoords(ti, tj, tk);

          os << "    {";
          os << "\"fn\":\"" << jsonEscape(fn.getName()) << "\",";
          os << "\"op\":\"" << jsonEscape(kind) << "\",";
          os << "\"owner_pu\":" << owner << ",";
          os << "\"ti\":" << ti << ",";
          os << "\"tj\":" << tj;
          if (tk >= 0)
            os << ",\"tk\":" << tk;
          extraFields();
          os << "}";
        };

        if (auto bc = dyn_cast<genmlir::gendram::BroadcastOp>(op)) {
          recordEvent(
              "broadcast",
              [&](int64_t &tiOut, int64_t &tjOut, int64_t &tkOut) {
                // Prefer explicit tile_* attributes if present.
                if (auto a = op->getAttrOfType<IntegerAttr>("tile_ti"))
                  tiOut = a.getInt();
                if (auto a = op->getAttrOfType<IntegerAttr>("tile_tj"))
                  tjOut = a.getInt();
                if (auto a = op->getAttrOfType<IntegerAttr>("tile_tk"))
                  tkOut = a.getInt();
              },
              [&] {
                // object + phase kind for broadcast tiles
                if (auto obj = op->getAttrOfType<StringAttr>("object"))
                  os << ",\"object\":\"" << jsonEscape(obj.getValue()) << "\"";
                if (auto pk = op->getAttrOfType<StringAttr>("phase_kind"))
                  os << ",\"phase_kind\":\"" << jsonEscape(pk.getValue()) << "\"";

                if (auto src = bc.getSrcPuAttr())
                  os << ",\"src_pu\":" << src.getInt();

                // If we know D + tile size, compute payload_bytes.
                if (dInfo && tileSize > 0) {
                  int64_t ti = -1, tj = -1, tk = -1;
                  if (auto a = op->getAttrOfType<IntegerAttr>("tile_ti"))
                    ti = a.getInt();
                  if (auto a = op->getAttrOfType<IntegerAttr>("tile_tj"))
                    tj = a.getInt();
                  if (ti >= 0 && tj >= 0) {
                    auto vv = tileView2D(ti, tj, tileSize, dInfo->dim0, dInfo->dim1);
                    int64_t bytes = vv.rows * vv.cols * dInfo->elemBytes;
                    os << ",\"payload_bytes\":" << bytes;
                  }
                }
              });
        } else if (auto s = dyn_cast<genmlir::gendram::SendOp>(op)) {
          recordEvent("send", [&](int64_t &, int64_t &, int64_t &) {}, [&] {
            os << ",\"qid\":" << s.getQid();

            // Optional payload size (helps timing for pipeline-mode).
            if (auto mty = dyn_cast<MemRefType>(s.getPayload().getType())) {
              if (mty.hasStaticShape()) {
                int64_t bytes = elemTypeBytes(mty.getElementType());
                for (int64_t d : mty.getShape())
                  bytes *= d;
                os << ",\"payload_bytes\":" << bytes;
              }
            }
          });
        } else if (auto r = dyn_cast<genmlir::gendram::RecvOp>(op)) {
          recordEvent("recv", [&](int64_t &, int64_t &, int64_t &) {}, [&] {
            os << ",\"qid\":" << r.getQid();
          });
        } else if (auto bar = dyn_cast<genmlir::gendram::BarrierOp>(op)) {
          recordEvent("barrier", [&](int64_t &, int64_t &, int64_t &) {}, [&] {
            if (auto scope = bar.getScopeAttr())
              os << ",\"scope\":\"" << jsonEscape(scope.getValue()) << "\"";
          });
        } else if (auto mt = dyn_cast<genmlir::gendram::MemTransferOp>(op)) {
          recordEvent("mem_transfer", [&](int64_t &, int64_t &, int64_t &) {}, [&] {
            if (auto k = mt.getKindAttr())
              os << ",\"kind\":\"" << jsonEscape(k.getValue()) << "\"";
          });
        }
      });
    });

    os << "\n  ]\n";
    os << "}\n";
  }
};

} // namespace

std::unique_ptr<Pass> createGenDramEmitArtifacts() {
  return std::make_unique<GenDramEmitArtifactsPass>();
}

std::unique_ptr<Pass>
createGenDramEmitArtifacts(const GenDramEmitArtifactsOptions &options) {
  return std::make_unique<GenDramEmitArtifactsPass>(options);
}

} // namespace genmlir
