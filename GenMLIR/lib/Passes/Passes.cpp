//===- Passes.cpp -------------------------------------------------------===//
//
// Pass registration.
//
// TableGen emits registerGenMLIRPasses() from include/genmlir/Passes/Passes.td.
// The -name GenMLIR option in the CMake file sets the function name.
//
//===----------------------------------------------------------------------===//

#include "genmlir/Passes/Passes.h"

namespace genmlir {
#define GEN_PASS_REGISTRATION
#include "genmlir/Passes/Passes.h.inc"
} // namespace genmlir
