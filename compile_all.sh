#!/bin/bash

set -euo pipefail

CLEAN_FIRST="${CLEAN_FIRST:-1}"
MACHINE="${machine:-${MACHINE:-}}"
ENABLE_CUDA="${ENABLE_CUDA:-1}"
ENABLE_OPENMP="${ENABLE_OPENMP:-1}"
NUM_OBJS_MIN="${NUM_OBJS_MIN:-3}"
NUM_OBJS_MAX="${NUM_OBJS_MAX:-7}"
MAKE_JOBS="${MAKE_JOBS:--j}"

if [[ "$ENABLE_CUDA" != "0" && "$ENABLE_CUDA" != "1" ]]; then
    echo "Error: ENABLE_CUDA must be 0 or 1"
    exit 1
fi

if [[ "$ENABLE_OPENMP" != "0" && "$ENABLE_OPENMP" != "1" ]]; then
    echo "Error: ENABLE_OPENMP must be 0 or 1"
    exit 1
fi

if [[ "$NUM_OBJS_MIN" -gt "$NUM_OBJS_MAX" ]]; then
    echo "Error: NUM_OBJS_MIN must be <= NUM_OBJS_MAX"
    exit 1
fi

MACHINE_ARGS=()
if [ -n "$MACHINE" ]; then
    MACHINE_ARGS=("machine=$MACHINE")
fi

BUILD_ARGS=(
    "ENABLE_CUDA=$ENABLE_CUDA"
    "ENABLE_OPENMP=$ENABLE_OPENMP"
)

NVCC_ARGS=()
if [ -n "${NVCC:-}" ]; then
    # Only override Makefile's auto-detection if user explicitly sets NVCC.
    NVCC_ARGS=("NVCC=$NVCC")
fi

if [ "$MACHINE" = "cc" ] && [ -z "${BOOST_ROOT:-}" ]; then
    echo "Error: BOOST_ROOT must be set for machine 'cc'"
    exit 1
fi

if [ "$CLEAN_FIRST" = "1" ]; then
    make "${MACHINE_ARGS[@]}" clean
fi

for i in $(seq "$NUM_OBJS_MIN" "$NUM_OBJS_MAX"); do
    make "${MACHINE_ARGS[@]}" "${NVCC_ARGS[@]}" "${BUILD_ARGS[@]}" "$MAKE_JOBS" NUM_OBJS="$i"
done
