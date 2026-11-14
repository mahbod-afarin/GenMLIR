//===- GenDRAMOps.h - GenDRAM dialect public header --------------------===//
//
// This header exposes the gendram dialect and op classes.
//
//===----------------------------------------------------------------------===//
#ifndef GENMLIR_DIALECT_GENDRAM_GENDRAMOPS_H
#define GENMLIR_DIALECT_GENDRAM_GENDRAMOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Dialect registration (generated)
#include "genmlir/Dialect/GenDRAM/GenDRAMOpsDialect.h.inc"

// Operation classes (generated)
#define GET_OP_CLASSES
#include "genmlir/Dialect/GenDRAM/GenDRAMOps.h.inc"

#endif // GENMLIR_DIALECT_GENDRAM_GENDRAMOPS_H
