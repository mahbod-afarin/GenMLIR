//===- DPGridOps.td ---------------------------------------------------===//
//
// The dp dialect of GenMLIR.
//
// It keeps the intent of a dynamic-programming kernel explicit in the IR:
// - the semi-ring of the update (min-plus for APSP, max-plus for alignment)
// - the schedule structure (phases for APSP, a wavefront for alignment)
// - tile markers with their coordinates
//
// Mid-end passes match these ops directly instead of recovering the pattern
// from arith and select ops.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/DialectBase.td"

def DPGrid_Dialect : Dialect {
  let name = "dp";
  let cppNamespace = "::genmlir::dp";
  let summary = "DP-level (semantic) dialect for GenMLIR";
  let description = [{
    The dp dialect captures dynamic-programming intent (semi-ring updates) and
    schedule markers (phases, tiling, wavefront) so that GenDRAM-aware passes
    can operate on stable IR patterns.
  }];
}

class DPGrid_Op<string mnemonic, list<Trait> traits = []>
    : Op<DPGrid_Dialect, mnemonic, traits>;

def DPGrid_YieldOp : DPGrid_Op<"yield", [Terminator, ReturnLike]> {
  let summary = "Terminator for dp region ops";
  let description = [{
    Terminator used by dp.tile, dp.phase, and dp.wavefront regions.
  }];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

// A single DP update step: "updated = semiring(current, candidate)".
// For min-plus: updated = min(current, candidate)
// For max-plus: updated = max(current, candidate)
//
// The semiring is carried as a string attribute:
//   semiring = "min_plus" or "max_plus"
def DPGrid_GridUpdateOp : DPGrid_Op<"grid_update",
    [Pure, SameOperandsAndResultType]> {
  let summary = "Semantic DP update (min-plus or max-plus)";
  let description = [{
    Represents one DP update, typically produced by recognizing a common
    compare/select pattern in low-level code. This keeps DP intent explicit
    and allows later passes to reason about semi-ring operations.

    Operands:
      - current: the existing value (e.g., D[i,j])
      - candidate: the new candidate (e.g., D[i,k] + D[k,j] or alignment score)

    Result:
      - updated value selected by the semi-ring (min_plus or max_plus)
  }];

  let arguments = (ins AnyType:$current, AnyType:$candidate, StrAttr:$semiring);
  let results = (outs AnyType:$updated);

  let assemblyFormat = "$current `,` $candidate attr-dict `:` type($updated)";
}

// A tile marker: a region that holds the computation of one tile. The tile
// coordinates (ti, tj, tk) and the owner PU (owner_pu) are attributes.
def DPGrid_TileOp : DPGrid_Op<"tile", [NoRegionArguments, RecursiveMemoryEffects]> {
  let summary = "Tile marker with coordinate metadata and a body region";
  let description = [{
    dp.tile is a structured marker op used to group the IR that computes a tile.
    The op carries logical coordinates (ti,tj[,tk]) so later passes can:
      - compute PU ownership (mapping),
      - derive wavefront steps, and
      - insert communication around tile boundaries.

    The op contains a single region whose terminator is dp.yield.
  }];

  let results = (outs);

  let regions = (region AnyRegion:$body);

  let arguments = (ins
    I64Attr:$ti,
    I64Attr:$tj,
    OptionalAttr<I64Attr>:$tk,
    OptionalAttr<I64Attr>:$owner_pu
  );

  let assemblyFormat = "attr-dict-with-keyword $body";
}

// Phases of blocked Floyd-Warshall (pivot / rowcol / internal).
def DPGrid_PhaseOp : DPGrid_Op<"phase", [NoRegionArguments, RecursiveMemoryEffects]> {
  let summary = "APSP phase marker (pivot, rowcol, internal)";
  let description = [{
    dp.phase groups the code of a blocked APSP phase and carries a 'kind'
    attribute:
      - kind = "pivot"
      - kind = "rowcol"
      - kind = "internal"
  }];

  let regions = (region AnyRegion:$body);
  let arguments = (ins StrAttr:$kind);
  let results = (outs);

  let assemblyFormat = "attr-dict-with-keyword $body";
}

// A wavefront marker for alignment-style DP.
// The wavefront traversal order is generally a diagonal schedule.
def DPGrid_WavefrontOp : DPGrid_Op<"wavefront", [NoRegionArguments, RecursiveMemoryEffects]> {
  let summary = "Wavefront schedule marker";
  let description = [{
    dp.wavefront groups the code that should follow a wavefront schedule.
    The actual ordering (e.g., by (ti+tj)) is produced by a mid-end pass
    which can annotate dp.tile ops with per-tile wavefront steps.
  }];

  let arguments = (ins);
  let results = (outs);

  let regions = (region AnyRegion:$body);

  let assemblyFormat = "attr-dict-with-keyword $body";
}