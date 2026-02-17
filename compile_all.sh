#!/bin/bash

set -euo pipefail

NVCC_BIN="${NVCC:-nvcc}"
CLEAN_FIRST="${CLEAN_FIRST:-1}"

if [ "$CLEAN_FIRST" = "1" ]; then
    make clean
fi

for i in $(seq 3 7); do
    make -j NUM_OBJS="$i" NVCC="$NVCC_BIN"
done
