//===- Passes.h - GenMLIR pass declarations ----------------------------===//
//
// This header exposes:
//  - pass class declarations generated from Passes.td
//  - pass registration entrypoint
//
// The actual pass implementations live in lib/Passes/*.
//
//===----------------------------------------------------------------------===//
#ifndef GENMLIR_PASSES_H
#define GENMLIR_PASSES_H

#include "mlir/Pass/Pass.h"

namespace genmlir {

// Generated pass declarations.
#define GEN_PASS_DECL
#include "genmlir/Passes/Passes.h.inc"

// Register all GenMLIR passes with the global MLIR pass registry.
void registerGenMLIRPasses();

} // namespace genmlir

#endif // GENMLIR_PASSES_H
