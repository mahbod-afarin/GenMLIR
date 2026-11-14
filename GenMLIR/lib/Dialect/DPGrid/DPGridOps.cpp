//===- DPGridOps.cpp ----------------------------------------------------===//
//
// Dialect and op definitions of the dp dialect. The code is produced by ODS
// from DPGridOps.td and included here.
//
//===----------------------------------------------------------------------===//

#include "genmlir/Dialect/DPGrid/DPGridOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace genmlir::dp;

//===----------------------------------------------------------------------===//
// Dialect registration (generated)
//===----------------------------------------------------------------------===//
#include "genmlir/Dialect/DPGrid/DPGridOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Op definitions (generated)
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "genmlir/Dialect/DPGrid/DPGridOps.cpp.inc"

void DPGridDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "genmlir/Dialect/DPGrid/DPGridOps.cpp.inc"
#undef GET_OP_LIST
  >();
}
