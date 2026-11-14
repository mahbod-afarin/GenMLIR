//===- genmlir-opt.cpp -------------------------------------------------===//
//
// mlir-opt style driver for the GenMLIR compiler.
//
// It registers:
//   - the dp and gendram dialects
//   - all GenMLIR passes
// and then delegates to MlirOptMain.
//
// Usage example:
//   genmlir-opt input.mlir --dp-extract --dp-canonicalize --gendram-mode-select ...
//
//===----------------------------------------------------------------------===//

#include "genmlir/Dialect/DPGrid/DPGridOps.h"
#include "genmlir/Dialect/GenDRAM/GenDRAMOps.h"
#include "genmlir/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace genmlir {
#define GEN_PASS_REGISTRATION
#include "genmlir/Passes/Passes.h.inc"
} // namespace genmlir

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;

  // Upstream dialects used by GenMLIR modules.
  registry.insert<func::FuncDialect,
                  scf::SCFDialect,
                  memref::MemRefDialect,
                  arith::ArithDialect>();

  registry.insert<genmlir::dp::DPGridDialect,
                  genmlir::gendram::GenDRAMDialect>();

  genmlir::registerGenMLIRPasses();

  return failed(MlirOptMain(argc, argv, "GenMLIR optimizer\n", registry));
}
