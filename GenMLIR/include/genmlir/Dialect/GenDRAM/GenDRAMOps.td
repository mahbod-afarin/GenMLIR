//===- GenDRAMOps.td --------------------------------------------------===//
//
// The gendram dialect of GenMLIR.
//
// It makes device-level actions explicit in the IR:
// - communication: broadcast, send/recv, barrier
// - memory movement markers
//
// The mid-end inserts these ops. The back-end turns them into schedule
// events or lowers them to runtime API calls.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GenDRAM_Dialect : Dialect {
  let name = "gendram";
  let cppNamespace = "::genmlir::gendram";
  let summary = "GenDRAM device-level dialect";
  let description = [{
    The gendram dialect models GenDRAM-relevant actions such as broadcast,
    queue send/recv, barriers, and explicit memory movement.

    After the mid-end, programs should have explicit gendram ops so the
    back-end can emit device artifacts (layout.json, schedule.json, PU programs).
  }];
}

class GenDRAM_Op<string mnemonic, list<Trait> traits = []>
    : Op<GenDRAM_Dialect, mnemonic, traits>;

// Broadcast a payload (typically a tile) to all relevant PUs.
def GenDRAM_BroadcastOp : GenDRAM_Op<"broadcast",
    [MemoryEffects<[MemRead]>]> {
  let summary = "Broadcast a payload (tile) to other PUs";
  let arguments = (ins AnyType:$payload, OptionalAttr<I64Attr>:$src_pu);
  let results = (outs);

  let assemblyFormat = "type($payload) $payload attr-dict";
}

// Queue-based send (producer).
def GenDRAM_SendOp : GenDRAM_Op<"send",
    [MemoryEffects<[MemRead]>]> {
  let summary = "Send a payload to a queue";
  let arguments = (ins AnyType:$payload, I64Attr:$qid);
  let results = (outs);

  let assemblyFormat = "type($payload) $payload attr-dict";
}

// Queue-based receive (consumer).
def GenDRAM_RecvOp : GenDRAM_Op<"recv",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Receive from a queue (synchronization point)";
  let arguments = (ins I64Attr:$qid);
  let results = (outs);

  let assemblyFormat = "attr-dict";
}

// Barrier / synchronization marker.
// 'scope' is optional (e.g., "phase", "kernel", "global").
def GenDRAM_BarrierOp : GenDRAM_Op<"barrier",
    [MemoryEffects<[]>]> {
  let summary = "Barrier synchronization";
  let arguments = (ins OptionalAttr<StrAttr>:$scope);
  let results = (outs);

  let assemblyFormat = "attr-dict";
}

// Explicit memory transfer marker (conceptual DRAM<->SRAM movement).
def GenDRAM_MemTransferOp : GenDRAM_Op<"mem_transfer",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Explicit memory transfer marker";
  let arguments = (ins AnyType:$src, AnyType:$dst, OptionalAttr<StrAttr>:$kind);
  let results = (outs);

  let assemblyFormat = "type($src) $src `,` type($dst) $dst attr-dict";
}
