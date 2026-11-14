#!/usr/bin/env bash
# Build GenMLIR against an existing LLVM/MLIR build and run the APSP and/or
# alignment example pipelines. Each pipeline emits:
#   - layout.json
#   - schedule.json
#   - pu_programs/*.txt
#
# Requirements:
#   - cmake >= 3.20
#   - ninja
#   - a C++17 compiler (clang or gcc)
#   - python3
#   - an LLVM build with MLIR enabled (LLVM_ENABLE_PROJECTS=mlir)
#
# Usage:
#   ./build_and_run.sh --llvm-build /path/to/llvm-build
#
# Or provide the CMake package dirs directly:
#   ./build_and_run.sh --mlir-dir /path/to/llvm-build/lib/cmake/mlir --llvm-dir /path/to/llvm-build/lib/cmake/llvm
#
# Or clone and build LLVM first (slow):
#   ./build_and_run.sh --fetch-llvm
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TYPE="${BUILD_TYPE:-Release}"
GENMLIR_BUILD_DIR="${GENMLIR_BUILD_DIR:-${ROOT_DIR}/build}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/out}"
MODE="${MODE:-both}"   # apsp | aln | both
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
TRACE_PASSES=0

LLVM_BUILD=""
LLVM_SRC=""
MLIR_DIR=""
LLVM_DIR=""

print_help() {
  cat <<EOF
GenMLIR build_and_run.sh

Options:
  --llvm-build <dir>     Path to an LLVM build directory (must contain lib/cmake/mlir and lib/cmake/llvm).
  --mlir-dir <dir>       Path to MLIR CMake package dir (e.g., <llvm-build>/lib/cmake/mlir).
  --llvm-dir <dir>       Path to LLVM CMake package dir (e.g., <llvm-build>/lib/cmake/llvm).
  --mode <apsp|aln|both> Which pipelines to run (default: both).
  --out <dir>            Output directory for artifacts (default: ./out).
  --build <dir>          Build directory (default: ./build).
  --build-type <type>    CMake build type (default: Release).
  --jobs <N>             Parallel build jobs (default: detected CPUs).
  --trace-passes         Print input/output IR for each genmlir-opt pass.

LLVM from source (optional):
  --fetch-llvm           Clone llvm-project into ./third_party/llvm-project and build it into ./third_party/llvm-build.

Examples:
  ./build_and_run.sh --llvm-build \$HOME/llvm/build
  ./build_and_run.sh --mlir-dir \$HOME/llvm/build/lib/cmake/mlir --llvm-dir \$HOME/llvm/build/lib/cmake/llvm
  MODE=apsp OUT_DIR=/tmp/out ./build_and_run.sh --llvm-build \$HOME/llvm/build
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

# Parse args
FETCH_LLVM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --llvm-build) LLVM_BUILD="${2:-}"; shift 2;;
    --mlir-dir) MLIR_DIR="${2:-}"; shift 2;;
    --llvm-dir) LLVM_DIR="${2:-}"; shift 2;;
    --mode) MODE="${2:-}"; shift 2;;
    --out) OUT_DIR="${2:-}"; shift 2;;
    --build) GENMLIR_BUILD_DIR="${2:-}"; shift 2;;
    --build-type) BUILD_TYPE="${2:-}"; shift 2;;
    --jobs) JOBS="${2:-}"; shift 2;;
    --fetch-llvm) FETCH_LLVM=1; shift 1;;
    --trace-passes) TRACE_PASSES=1; shift 1;;
    -h|--help) print_help; exit 0;;
    *) die "unknown option: $1 (use --help)";;
  esac
done

run_genmlir_passes() {
  local input_file="$1"
  local output_file="$2"
  local pipeline_name="$3"
  shift 3
  local passes=("$@")

  if [[ ${TRACE_PASSES} -eq 0 ]]; then
    "${GENMLIR_OPT}" "${input_file}" "${passes[@]}" -o "${output_file}"
    return 0
  fi

  local current_file="${input_file}"
  local pass_index=1
  local pass_dir="${OUT_DIR}/pass-dumps/${pipeline_name}"
  mkdir -p "${pass_dir}"

  for pass_arg in "${passes[@]}"; do
    local pass_name="${pass_arg%%=*}"
    pass_name="${pass_name##--}"
    local safe_pass_name="${pass_name//[^A-Za-z0-9._-]/_}"
    local step_file="${pass_dir}/$(printf "%02d" "${pass_index}")_${safe_pass_name}.mlir"

    echo "[pass:${pipeline_name}] [${pass_index}] running ${pass_arg}"
    echo "  input -> ${current_file}"
    echo "  output -> ${step_file}"
    echo "--- input MLIR (${pipeline_name} #${pass_index}: ${pass_arg}) ---"
    cat "${current_file}"
    echo "--- end input ---"

    "${GENMLIR_OPT}" "${current_file}" "${pass_arg}" -o "${step_file}"

    echo "--- output MLIR (${pipeline_name} #${pass_index}: ${pass_arg}) ---"
    cat "${step_file}"
    echo "--- end output ---"

    current_file="${step_file}"
    ((pass_index += 1))
  done

  cp "${current_file}" "${output_file}"
}

# Validate mode
case "${MODE}" in
  apsp|aln|both) ;;
  *) die "--mode must be one of: apsp, aln, both (got: ${MODE})";;
esac

mkdir -p "${OUT_DIR}"

# Optionally fetch/build LLVM
if [[ "${FETCH_LLVM}" -eq 1 ]]; then
  LLVM_SRC="${ROOT_DIR}/third_party/llvm-project"
  LLVM_BUILD="${ROOT_DIR}/third_party/llvm-build"
  if [[ ! -d "${LLVM_SRC}/.git" ]]; then
    echo "[1/4] Cloning llvm-project into ${LLVM_SRC}"
    mkdir -p "${ROOT_DIR}/third_party"
    git clone --depth=1 https://github.com/llvm/llvm-project.git "${LLVM_SRC}"
  else
    echo "[1/4] llvm-project already exists at ${LLVM_SRC}"
  fi

  echo "[2/4] Configuring LLVM/MLIR build in ${LLVM_BUILD}"
  cmake -G Ninja -S "${LLVM_SRC}/llvm" -B "${LLVM_BUILD}" \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=Native \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DLLVM_ENABLE_ASSERTIONS=ON

  echo "[3/4] Building LLVM/MLIR (this can take a while)"
  ninja -C "${LLVM_BUILD}" -j "${JOBS}" mlir-opt
  echo "[4/4] LLVM/MLIR build complete"
fi

# Derive MLIR_DIR/LLVM_DIR from --llvm-build if needed
if [[ -n "${LLVM_BUILD}" ]]; then
  MLIR_DIR="${MLIR_DIR:-${LLVM_BUILD}/lib/cmake/mlir}"
  LLVM_DIR="${LLVM_DIR:-${LLVM_BUILD}/lib/cmake/llvm}"
fi

[[ -n "${MLIR_DIR}" ]] || die "MLIR_DIR not set. Pass --llvm-build or --mlir-dir."
[[ -n "${LLVM_DIR}" ]] || die "LLVM_DIR not set. Pass --llvm-build or --llvm-dir."
[[ -d "${MLIR_DIR}" ]] || die "MLIR_DIR does not exist: ${MLIR_DIR}"
[[ -d "${LLVM_DIR}" ]] || die "LLVM_DIR does not exist: ${LLVM_DIR}"

# Build GenMLIR
echo "[build] Configuring GenMLIR in ${GENMLIR_BUILD_DIR}"
cmake -G Ninja -S "${ROOT_DIR}" -B "${GENMLIR_BUILD_DIR}" \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

echo "[build] Building genmlir-opt"
ninja -C "${GENMLIR_BUILD_DIR}" -j "${JOBS}" genmlir-opt

GENMLIR_OPT="${GENMLIR_BUILD_DIR}/bin/genmlir-opt"
[[ -x "${GENMLIR_OPT}" ]] || die "genmlir-opt not found/executable at: ${GENMLIR_OPT}"

# Generate MLIR from GenDSL
APSP_MLIR="${OUT_DIR}/apsp.mlir"
ALN_MLIR="${OUT_DIR}/alignment.mlir"

if [[ "${MODE}" == "apsp" || "${MODE}" == "both" ]]; then
  echo "[run] Generating APSP MLIR from GenDSL"
  python3 "${ROOT_DIR}/tools/gendsl-translate.py" "${ROOT_DIR}/examples/gendsl_apsp.yaml" > "${APSP_MLIR}"

  echo "[run] Running APSP pipeline -> ${OUT_DIR}/genmlir_apsp"
  run_genmlir_passes "${APSP_MLIR}" "${OUT_DIR}/apsp.lowered.mlir" "apsp" \
    --dp-canonicalize \
    --gendram-mode-select \
    --gendram-mapping="C=4 G=8 M=64" \
    --gendram-tiering \
    --gendram-comm-insert \
    --gendram-emit-artifacts="out_dir=${OUT_DIR}/genmlir_apsp"
fi

if [[ "${MODE}" == "aln" || "${MODE}" == "both" ]]; then
  echo "[run] Generating Alignment MLIR from GenDSL"
  python3 "${ROOT_DIR}/tools/gendsl-translate.py" "${ROOT_DIR}/examples/gendsl_alignment.yaml" > "${ALN_MLIR}"

  echo "[run] Running Alignment pipeline -> ${OUT_DIR}/genmlir_alignment"
  run_genmlir_passes "${ALN_MLIR}" "${OUT_DIR}/alignment.lowered.mlir" "alignment" \
    --dp-canonicalize \
    --gendram-mode-select \
    --gendram-mapping="C=4 G=8 M=64" \
    --gendram-tiering \
    --gendram-comm-insert \
    --gendram-emit-artifacts="out_dir=${OUT_DIR}/genmlir_alignment"
fi

echo
echo "Done."
echo "Artifacts written under:"
if [[ "${MODE}" == "apsp" || "${MODE}" == "both" ]]; then
  echo "  ${OUT_DIR}/genmlir_apsp/"
fi
if [[ "${MODE}" == "aln" || "${MODE}" == "both" ]]; then
  echo "  ${OUT_DIR}/genmlir_alignment/"
fi
