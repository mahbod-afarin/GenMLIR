add_mlir_dialect_library(GenMLIRDPGrid
  DPGridOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/genmlir/Dialect/DPGrid

  DEPENDS
    MLIRDPGridOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRMemRefDialect
)
