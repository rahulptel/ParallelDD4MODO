#!/bin/bash

set -euo pipefail

NVCC_BIN="${NVCC:-nvcc}"
CLEAN_FIRST="${CLEAN_FIRST:-1}"
MACHINE="${machine:-${MACHINE:-}}"
MACHINE_ARGS=()
if [ -n "$MACHINE" ]; then
    MACHINE_ARGS=("machine=$MACHINE")
fi

if [ "$MACHINE" = "cc" ] && [ -z "${BOOST_ROOT:-}" ]; then
    echo "Error: BOOST_ROOT must be set for machine 'cc'"
    exit 1
fi

if [ "$CLEAN_FIRST" = "1" ]; then
    make "${MACHINE_ARGS[@]}" clean
fi

for i in $(seq 3 7); do
    make "${MACHINE_ARGS[@]}" -j NUM_OBJS="$i" NVCC="$NVCC_BIN"
done
