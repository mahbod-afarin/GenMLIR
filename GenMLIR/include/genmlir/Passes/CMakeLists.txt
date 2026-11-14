set(LLVM_TARGET_DEFINITIONS Passes.td)
# The '-name' controls the generated registration function name:
#   void registerGenMLIRPasses();
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GenMLIR)
add_public_tablegen_target(GenMLIRPassIncGen)

add_mlir_doc(GenMLIRPasses GenMLIRPasses Passes/ -gen-pass-doc)
