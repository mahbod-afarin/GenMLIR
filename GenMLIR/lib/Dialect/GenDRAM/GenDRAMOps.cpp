//===- GenDRAMOps.cpp ---------------------------------------------------===//
//
// Dialect and op definitions of the gendram dialect. The code is produced by
// ODS from GenDRAMOps.td and included here.
//
//===----------------------------------------------------------------------===//

#include "genmlir/Dialect/GenDRAM/GenDRAMOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace genmlir::gendram;

//===----------------------------------------------------------------------===//
// Dialect registration (generated)
//===----------------------------------------------------------------------===//
#include "genmlir/Dialect/GenDRAM/GenDRAMOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Op definitions (generated)
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "genmlir/Dialect/GenDRAM/GenDRAMOps.cpp.inc"

void GenDRAMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "genmlir/Dialect/GenDRAM/GenDRAMOps.cpp.inc"
#undef GET_OP_LIST
  >();
}
