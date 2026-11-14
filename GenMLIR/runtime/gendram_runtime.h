//===- gendram_runtime.h ------------------------------------------------===//
//
// GenDRAM runtime API.
//
// The back-end pass --gendram-lower-to-runtime-calls lowers gendram dialect
// ops to calls to these functions. On a device they would enqueue commands,
// manage the queues and synchronize the PUs. The implementation in
// gendram_runtime.cpp prints a trace of the calls instead.
//
//===----------------------------------------------------------------------===//
#ifndef GENMLIR_RUNTIME_GENDRAM_RUNTIME_H
#define GENMLIR_RUNTIME_GENDRAM_RUNTIME_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Same layout as UnrankedMemRefType<float> of the MLIR ExecutionEngine
// (RunnerUtils), so that f32 memrefs of any rank can be passed.
struct UnrankedMemRefTypeF32 {
  int64_t rank;
  void *descriptor;
};

void gendram_broadcast(UnrankedMemRefTypeF32 payload);
void gendram_send(int64_t qid, UnrankedMemRefTypeF32 payload);
void gendram_recv(int64_t qid);
void gendram_barrier();
void gendram_mem_transfer(UnrankedMemRefTypeF32 src, UnrankedMemRefTypeF32 dst);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // GENMLIR_RUNTIME_GENDRAM_RUNTIME_H
