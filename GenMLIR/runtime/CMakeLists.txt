add_library(GenDRAMRuntime
  gendram_runtime.cpp
)

target_include_directories(GenDRAMRuntime PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime
)

target_link_libraries(GenDRAMRuntime PUBLIC
  MLIRExecutionEngine
)
