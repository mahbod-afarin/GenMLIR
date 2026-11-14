add_mlir_library(GenMLIRPasses
  Passes.cpp
  DpExtractPass.cpp
  DpCanonicalizePass.cpp
  DpTilingPass.cpp
  ApspPhaseFormationPass.cpp
  WavefrontSchedulePass.cpp
  GenDramModeSelectPass.cpp
  GenDramMappingPass.cpp
  GenDramTieringPass.cpp
  GenDramCommInsertPass.cpp
  GenDramLowerToRuntimeCallsPass.cpp
  GenDramEmitArtifactsPass.cpp

  DEPENDS
    GenMLIRPassIncGen

  LINK_LIBS PUBLIC
    GenMLIRDPGrid
    GenMLIRGenDRAM
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRArithDialect
)
