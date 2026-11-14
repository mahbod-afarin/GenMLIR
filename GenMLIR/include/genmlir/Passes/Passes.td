//===- Passes.td - GenMLIR Passes -------------------------*- tablegen -*-===//
//
// Pass definitions of the GenMLIR compiler, by pass group:
//
//  Group 1: DP recognition/normalization
//    - dp-extract
//    - dp-canonicalize
//
//  Group 2: Tiling & schedule formation
//    - dp-tiling
//    - apsp-phase-formation
//    - wavefront-schedule
//
//  Group 3: Mapping & memory placement
//    - gendram-mode-select
//    - gendram-mapping
//    - gendram-tiering
//
//  Group 4: Communication insertion
//    - gendram-comm-insert
//
//  Back-end / artifacts
//    - gendram-lower-to-runtime-calls
//    - gendram-emit-artifacts
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def DpExtract : Pass<"dp-extract", "mlir::func::FuncOp"> {
  let summary = "Recognize DP compare/select patterns and create dp.grid_update ops";
  let description = [{
    Replaces select(cmp(candidate, current), candidate, current) with a
    dp.grid_update op annotated with semiring=min_plus or semiring=max_plus.
    Other shapes are left unchanged.
  }];
}

def DpCanonicalize : Pass<"dp-canonicalize", "mlir::func::FuncOp"> {
  let summary = "Normalize the semiring attribute of dp.grid_update";
  let description = [{
    Checks the semiring attribute of every dp.grid_update and sets it to
    min_plus if it is missing or unknown.
  }];
}

def DpTiling : Pass<"dp-tiling", "mlir::func::FuncOp"> {
  let summary = "Insert a dp.tile marker where none is present";
  let description = [{
    If a function contains no dp.tile, wraps its body in a single dp.tile
    with ti = tj = 0. Functions that already contain dp.tile ops, such as
    those emitted by the GenDSL translator, are left unchanged.
  }];
  let options = [
    Option<"tileSize", "tile-size", "int", /*default=*/"64",
           "Logical tile size B (currently unused)">
  ];
}

def ApspPhaseFormation : Pass<"apsp-phase-formation", "mlir::func::FuncOp"> {
  let summary = "Ensure APSP code is grouped into dp.phase (pivot/rowcol/internal)";
  let description = [{
    If a function contains no dp.phase, wraps its body in a single
    dp.phase(kind="single"). Functions that already contain dp.phase ops are
    left unchanged.
  }];
}

def WavefrontSchedule : Pass<"wavefront-schedule", "mlir::func::FuncOp"> {
  let summary = "Annotate dp.tile ops with wavefront step metadata";
  let description = [{
    For alignment-style DP, tiles follow a wavefront order. This pass annotates
    each dp.tile with a derived integer 'wavefront_step' attribute (ti+tj),
    which can be used by back-end scheduling or artifact emission.
  }];
}

def GenDramModeSelect : Pass<"gendram-mode-select", "mlir::ModuleOp"> {
  let summary = "Select GenDRAM execution mode and record it as module attributes";
  let description = [{
    Sets module attribute gendram.mode to either:
      - "broadcast"  (APSP-style)
      - "pipeline"   (genomics-style)

    Pipeline is selected if the module has no dp.phase and contains functions
    named seeding and alignment. Otherwise broadcast is selected. An existing
    gendram.mode attribute is kept.
  }];
}

def GenDramMapping : Pass<"gendram-mapping", "mlir::func::FuncOp"> {
  let summary = "Assign dp.tile operations to owner PUs using a modulo mapping";
  let description = [{
    Annotates dp.tile with owner_pu = (ti*M + tj) mod (C*G).

    Options:
      - C: channels
      - G: bank-groups per channel
      - M: matrix tiles per row (or a logical stride; see paper)
  }];
  let options = [
    Option<"C", "C", "int", /*default=*/"4", "Number of channels">,
    Option<"G", "G", "int", /*default=*/"8", "Bank-groups per channel">,
    Option<"M", "M", "int", /*default=*/"64", "Logical tiles-per-row factor used in mapping">
  ];
}

def GenDramTiering : Pass<"gendram-tiering", "mlir::ModuleOp"> {
  let summary = "Assign tier/channel/bank-group metadata to memory objects";
  let description = [{
    Placement policy:
      - Objects whose name contains 'PTR' or 'CAL' go to tier 0.
      - All other objects go to tier 1.
      - Objects with at least one million elements are marked as interleaved
        across channels and bank-groups.

    The result is encoded as attributes on memref.alloc/memref.global ops.
  }];
}

def GenDramCommInsert : Pass<"gendram-comm-insert", "mlir::ModuleOp"> {
  let summary = "Insert gendram broadcast/barrier or send/recv based on mode";
  let description = [{
    Inserts explicit communication:
      - In broadcast mode: a broadcast of the pivot tile after every
        dp.phase(kind="pivot"), a broadcast of every row and column tile
        after every dp.phase(kind="rowcol"), and a barrier after every phase.
      - In pipeline mode: gendram.send at the end of the seeding kernel and
        gendram.recv at the start of the alignment kernel, both on queue 0.
  }];
}

def GenDramLowerToRuntimeCalls : Pass<"gendram-lower-to-runtime-calls", "mlir::ModuleOp"> {
  let summary = "Lower gendram.* ops to func.call of runtime API stubs";
  let description = [{
    Converts gendram.broadcast/send/recv/barrier/mem_transfer into calls to
    runtime functions declared in runtime/gendram_runtime.h.

    The runtime API takes payloads as unranked f32 memrefs. An op whose
    payload has another element type is left in place.
  }];
}

def GenDramEmitArtifacts : Pass<"gendram-emit-artifacts", "mlir::ModuleOp"> {
  let summary = "Emit layout.json, schedule.json, and pu_programs/*";
  let description = [{
    Writes the compiler artifacts:
      - Memory layout plan (layout.json)
      - Communication/synchronization schedule (schedule.json)
      - PU programs (pu_programs/*.txt)

    The pass scans the IR for:
      - placement attrs on memref objects (tier/channel/bank-group/interleave)
      - dp.tile owner_pu attrs
      - gendram communication ops in program order
  }];
  let options = [
    Option<"outDir", "out_dir", "std::string", "\"./gendram_out\"",
           "Output directory for artifacts">
  ];
}
