# Parallel Decision Diagram-based Multiobjective Discrete Optimization

Parallel Decision Diagram-based Multiobjective Discrete Optimization, or
`PDDMODO`, is a C++ research codebase for exact multiobjective optimization
with decision diagrams. It builds binary decision diagrams (BDDs) or multivalued
decision diagrams (MDDs) for supported problem classes, then enumerates the
Pareto frontier with CPU algorithms and selected CUDA implementations.

The main executable is compiled for a fixed number of objectives:

```bash
multiobj_nobjs<NUM_OBJS>
```

For example, a 3-objective build produces `multiobj_nobjs3`. Use a binary whose
`NUM_OBJS` matches the objective count in the input instance.

## Project Layout

```text
.
|-- makefile              # Main build file
|-- compile_all.sh        # Helper script for building several NUM_OBJS binaries
|-- src/
|   |-- main.cpp          # CLI entry point, instance dispatch, build/enumeration flow
|   |-- bdd/              # BDD data structures and constructors (knapsack, independent set)
|   |-- mdd/              # MDD data structures and TSP MDD constructor
|   |-- instances/        # Input parsers for knapsack, set packing, independent set, TSP
|   |-- enum/             # Pareto frontier algorithms and enumeration dispatch
|   |   |-- cpu/          # CPU-based BFS, bottom-up, and layer cutset coupling algorithms
|   |   |-- gpu/          # CUDA kernels/wrappers for GPU-based BFS and dynamic layer cutset
|   |   |-- pareto_frontier.hpp  # Nondominated frontier container and merge/convolution logic
|   |   `-- multiobj_enum.hpp/.cpp # Central dispatch hub for CPU/GPU frontier enumeration
|   `-- util/             # CLI parsing, output, stats, OpenMP helpers, CPU affinity, GPU options
|-- data/                 # Benchmark/test instances grouped by objective count
|-- cc/                   # Cluster/experiment job scripts and tables
`-- results/              # Result summaries and plotting scripts, when present
```

The currently wired problem types are:

- `1`: multiobjective knapsack, represented as a BDD.
- `2`: multiobjective set packing, converted to an independent-set BDD.
- `3`: multiobjective TSP, represented as an MDD.

The available enumeration methods are:

- `1`: top-down BFS frontier propagation.
- `2`: bottom-up BFS frontier propagation.
- `3`: dynamic layer cutset coupling.

Pareto frontier enumeration code is located in `src/enum/`:

- `src/enum/pareto_frontier.hpp`: Pareto frontier data structure, merge, and convolution logic.
- `src/enum/multiobj_enum.hpp/.cpp`: central interface/dispatch for CPU and GPU enumeration methods.

CPU enumeration methods (`src/enum/cpu/`):
- `enum.cpp`: BDD/MDD top-down and bottom-up BFS frontier propagation.
- `couple.cpp`: BDD/MDD dynamic layer cutset coupling.
- `dominance.cpp`: BDD knapsack/set packing state dominance filtering.
- `bottomup.cpp`: bottom-up enumeration helpers.
- `topdown.cpp`: top-down enumeration helpers.
- `cpu_helpers.hpp`: shared CPU functions for frontier operations.
- `cpu_wrappers.hpp`: CPU method declarations.

GPU/CUDA enumeration methods (`src/enum/gpu/`):
- `enum.cu`: CUDA wrappers, device setup, and execution dispatch.
- `topdown.cu`: BDD and MDD GPU top-down frontier propagation.
- `bottomup.cu`: GPU bottom-up propagation kernels/helpers.
- `couple.cu`: MDD dynamic layer cutset coupling kernels (cutset product join/merge).
- `enum_types.cuh` and `dominance_utils.cuh`: CUDA-specific type definitions and device utility functions.
- `cuda_stubs.cpp`: Stub implementations used during CPU-only builds (`ENABLE_CUDA=0`).
- `cuda_wrappers.hpp`: GPU method declarations.

## Build

Builds are controlled by the root `makefile`. The important options are:

- `NUM_OBJS=<N>`: compile-time objective dimension. Default: `3`.
- `ENABLE_CUDA=0|1`: include CUDA kernels. Default: `1`.
- `ENABLE_OPENMP=0|1`: include OpenMP CPU parallelism. Default: `1`.
- `machine=cc`: use `BOOST_ROOT` as the Boost location. This can come from
  the environment as well as the make command line.
- `NVCC=<path>`: override CUDA compiler auto-detection.

The makefile uses `g++` for C++ sources and requires Boost headers under
`/opt/boost/include` by default. CUDA builds require `nvcc >= 12`.

If Boost is installed elsewhere, pass the Boost prefix explicitly:

```bash
make BOOSTDIR=/path/to/boost-prefix NUM_OBJS=3 ENABLE_CUDA=0 ENABLE_OPENMP=1
```

### CPU-Only Build

Use this when CUDA is unavailable or when you only want CPU execution:

```bash
make clean
make NUM_OBJS=3 ENABLE_CUDA=0 ENABLE_OPENMP=1
```

For a serial CPU build without OpenMP:

```bash
make clean
make NUM_OBJS=3 ENABLE_CUDA=0 ENABLE_OPENMP=0
```

When OpenMP is disabled, explicit CPU thread arguments are rejected at runtime
and the binary runs with one CPU thread.

### GPU Build

Use this when `nvcc` and the CUDA runtime are available:

```bash
make clean
make NUM_OBJS=3 ENABLE_CUDA=1 ENABLE_OPENMP=1
```

If `nvcc` is not on the default path:

```bash
make clean
make NUM_OBJS=3 ENABLE_CUDA=1 ENABLE_OPENMP=1 NVCC=/usr/local/cuda/bin/nvcc
```

### Build Multiple Objective Dimensions

`compile_all.sh` builds `multiobj_nobjs3` through `multiobj_nobjs7` by default:

```bash
./compile_all.sh
```

Useful overrides:

```bash
ENABLE_CUDA=0 ENABLE_OPENMP=1 ./compile_all.sh
NUM_OBJS_MIN=4 NUM_OBJS_MAX=6 ./compile_all.sh
CLEAN_FIRST=0 MAKE_JOBS=-j8 ./compile_all.sh
```

On Compute Canada-style environments using `machine=cc`, set `BOOST_ROOT`:

```bash
BOOST_ROOT=/path/to/boost machine=cc ENABLE_CUDA=0 ./compile_all.sh
```

If your shell already has `machine=cc` set, regular `make` commands also need
`BOOST_ROOT`:

```bash
BOOST_ROOT=/path/to/boost make NUM_OBJS=3 ENABLE_CUDA=0 ENABLE_OPENMP=1
```

## Run

General CLI:

```bash
./multiobj_nobjs3 <input-file> <problem-type> <method> <state_dominance> [options]
```

Arguments:

- `<input-file>`: path to an instance under `data/` or another compatible file.
- `<problem-type>`: `1` knapsack, `2` set packing, `3` TSP.
- `<method>`: `1` top-down, `2` bottom-up, `3` dynamic layer cutset.
- `<state_dominance>`: `0` disabled, `1` enabled where implemented.

Backend options:

```bash
--backend cpu|gpu
cpu [num_threads]
gpu [3]
```

CPU options:

```bash
--cpu-threads <N>
--cpu-kernel <1|3>
```

If `--cpu-threads` is omitted in an OpenMP build, the program uses
`OMP_NUM_THREADS` when it is a valid positive integer, otherwise `1`.

GPU options:

```bash
--kernel 3
```

The token `cuda` is intentionally rejected; use `gpu`.

Output options:

```bash
--save-frontier
--frontier-out <path>
--save-stats
--stats-out <path>
```

`--save-frontier` writes a gzip-compressed CSV frontier to
`<input_stem>.frontier.csv.gz` unless `--frontier-out` is provided.
`--save-stats` appends one JSONL record to `<input_stem>.stats.jsonl` unless
`--stats-out` is provided. Passing `--frontier-out` or `--stats-out` implies the
corresponding save option.

### Supported Backend/Method Combinations

For knapsack and set packing BDDs:

- CPU supports methods `1`, `2`, and `3`.
- GPU supports method `1`.
- GPU methods `2` and `3` fail fast as unsupported.

For TSP MDDs:

- CPU supports methods `1` and `3`.
- GPU supports methods `1` and `3`.
- Method `2` is not accepted for TSP.

## Test on a Small Instance

The commands below use a 3-objective knapsack instance included in `data/`.

### CPU Top-Down Test

```bash
make clean
make NUM_OBJS=3 ENABLE_CUDA=0 ENABLE_OPENMP=1

./multiobj_nobjs3 data/3/knapsack/KP_p-3_n-10_ins-1.dat 1 1 0 \
  --backend cpu --cpu-threads 4 --cpu-kernel 1 \
  --save-stats --stats-out test.cpu.stats.jsonl
```

### CPU Dynamic Layer Cutset Test

```bash
./multiobj_nobjs3 data/3/knapsack/KP_p-3_n-10_ins-1.dat 1 3 0 \
  --backend cpu --cpu-threads 4 --cpu-kernel 3 \
  --save-frontier --frontier-out test.cpu.frontier.csv.gz
```

### GPU Top-Down Test

```bash
make clean
make NUM_OBJS=3 ENABLE_CUDA=1 ENABLE_OPENMP=1

./multiobj_nobjs3 data/3/knapsack/KP_p-3_n-10_ins-1.dat 1 1 0 \
  --backend gpu --kernel 3 \
  --save-stats --stats-out test.gpu.stats.jsonl
```

### TSP MDD Test

```bash
./multiobj_nobjs3 data/3/tsp/tsp-nobj3-ncities5-seed495.dat 3 1 0 \
  --backend cpu --cpu-threads 4 --cpu-kernel 1
```

Successful runs always print three lines:

1. Number of Pareto solutions.
2. CPU total time, equal to compile time plus enumeration time.
3. Tab-separated run statistics. BDD problem types include BDD structure fields;
   TSP prints compile/enumeration timing fields.

When comparing CPU and GPU runs, the first line should match for the same
instance, problem type, method semantics, and objective count.
