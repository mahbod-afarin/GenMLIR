add_executable(genmlir-opt
  genmlir-opt.cpp
)

target_link_libraries(genmlir-opt PRIVATE
  GenMLIRDPGrid
  GenMLIRGenDRAM
  GenMLIRPasses

  MLIRIR
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIROptLib
)

mlir_check_all_link_libraries(genmlir-opt)
