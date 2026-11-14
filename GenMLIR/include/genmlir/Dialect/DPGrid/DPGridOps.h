//===- DPGridOps.h - DP dialect public header --------------------------===//
//
// This header exposes the dp dialect and op classes.
//
// Generated declarations are included via the *.inc files produced by ODS.
//
//===----------------------------------------------------------------------===//
#ifndef GENMLIR_DIALECT_DPGRID_DPGRIDOPS_H
#define GENMLIR_DIALECT_DPGRID_DPGRIDOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Dialect registration (generated)
#include "genmlir/Dialect/DPGrid/DPGridOpsDialect.h.inc"

// Operation classes (generated)
#define GET_OP_CLASSES
#include "genmlir/Dialect/DPGrid/DPGridOps.h.inc"

#endif // GENMLIR_DIALECT_DPGRID_DPGRIDOPS_H
