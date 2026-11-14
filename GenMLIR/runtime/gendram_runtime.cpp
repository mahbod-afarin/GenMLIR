//===- gendram_runtime.cpp ----------------------------------------------===//
//
// Trace implementation of the runtime API. Every call prints its arguments to
// stderr and does nothing else. A simulator back-end would replace these
// bodies.
//
//===----------------------------------------------------------------------===//
#include "gendram_runtime.h"

#include <cstdio>

extern "C" void gendram_broadcast(UnrankedMemRefTypeF32 payload) {
  std::fprintf(stderr, "[gendram_rt] broadcast(payload_rank=%ld)\n",
               (long)payload.rank);
}

extern "C" void gendram_send(int64_t qid, UnrankedMemRefTypeF32 payload) {
  std::fprintf(stderr, "[gendram_rt] send(qid=%ld, payload_rank=%ld)\n",
               (long)qid, (long)payload.rank);
}

extern "C" void gendram_recv(int64_t qid) {
  std::fprintf(stderr, "[gendram_rt] recv(qid=%ld)\n", (long)qid);
}

extern "C" void gendram_barrier() {
  std::fprintf(stderr, "[gendram_rt] barrier()\n");
}

extern "C" void gendram_mem_transfer(UnrankedMemRefTypeF32 src,
                                     UnrankedMemRefTypeF32 dst) {
  std::fprintf(stderr,
               "[gendram_rt] mem_transfer(src_rank=%ld, dst_rank=%ld)\n",
               (long)src.rank, (long)dst.rank);
}
