add_mlir_dialect_library(GenMLIRGenDRAM
  GenDRAMOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/genmlir/Dialect/GenDRAM

  DEPENDS
    MLIRGenDRAMOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRMemRefDialect
)
