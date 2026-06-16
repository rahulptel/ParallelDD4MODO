# cuMODD Agent Guide

## 0) Working Style
- Be direct, technical, and concise. This is research code, not product code.
- Make the smallest code change that solves the requested problem.
- Prefer readable C++ over clever abstractions. Match local style even when it is imperfect.
- If a request is ambiguous in a way that changes the implementation, ask before editing.
- Do not refactor adjacent code, reformat unrelated files, or add speculative features.
- If you notice unrelated dead code or cleanup opportunities, mention them instead of silently changing them.

## 1) Repository Purpose
- This is a C++ decision-diagram codebase for multiobjective optimization.
- Main executable pattern: `multiobj_nobjs<NUM_OBJS>`, for example `multiobj_nobjs3`.
- Core idea: build an exact BDD or MDD, then enumerate the Pareto frontier by dynamic-programming style frontier propagation.
- Supported problem types in the current `src/main.cpp`:
  - `1`: Knapsack, represented with a BDD.
  - `2`: Set packing, converted to an independent-set BDD.
  - `3`: TSP, represented with an MDD.
- Older problem types from related branches, such as set covering, portfolio, and absolute-value models, are not wired into this repo.

## 2) Build and Environment
- Build system: root `makefile`.
- Host compiler: `g++` with C++11 flags.
- CUDA compiler: `nvcc` when `ENABLE_CUDA=1`; CUDA builds require detected `nvcc >= 12`.
- Boost headers are expected under `/opt/boost/include` by default.
- On `machine=cc`, `BOOSTDIR` is taken from `BOOST_ROOT`.
- Gurobi include/library settings exist in the makefile but are commented out.
- CPLEX/CP Optimizer are not linked in this cleaned branch.
- Objective dimension is compile-time:
  - `NUM_OBJS` defines macro `NOBJS`.
  - Input files may carry an objective count, but core containers and loops assume the binary was built with the matching `NOBJS`.

Common commands:
- `make NUM_OBJS=3`
- `make NUM_OBJS=3 ENABLE_CUDA=0`
- `make NUM_OBJS=3 ENABLE_OPENMP=1`
- `make clean`
- `make -n ENABLE_CUDA=0 NUM_OBJS=3`
- `./compile_all.sh`

`compile_all.sh` defaults:
- Builds `NUM_OBJS=3..7`.
- Uses `ENABLE_CUDA=1` and `ENABLE_OPENMP=1` unless overridden.
- Runs `make clean` first unless `CLEAN_FIRST=0`.
- Produces `multiobj_nobjs3`, `multiobj_nobjs4`, etc.

## 3) Runtime Interface
Usage:

```bash
./multiobj_nobjs3 <input-file> <problem-type> <method> <state_dominance> [options]
```

Problem types:
- `1`: knapsack.
- `2`: set packing.
- `3`: TSP.

Methods:
- `1`: top-down BFS frontier propagation.
- `2`: bottom-up BFS frontier propagation.
- `3`: dynamic layer cutset coupling.

Backend options:
- Default backend is CPU.
- Named form: `--backend cpu|gpu`.
- Shorthand form: `cpu [num_threads]` or `gpu [3]`.
- `cuda` is rejected; use `gpu`.
- `--cpu-threads <N>` requires an OpenMP-enabled build.
- `--cpu-kernel <1|3>` selects the CPU kernel variant for methods `1` and `3`.
- `--kernel 3` is accepted for GPU compatibility only.

Output options:
- `--save-frontier` writes `<input_stem>.frontier.csv.gz`.
- `--frontier-out <path>` writes a gzip-compressed CSV frontier to the explicit path.
- `--save-stats` appends one JSONL stats record.
- `--stats-out <path>` implies `--save-stats`; default is `<input_stem>.stats.jsonl`.

GPU support in current dispatch:
- BDD knapsack/set-packing: GPU is implemented for method `1` only.
- BDD method `2` and BDD method `3` reject GPU.
- TSP/MDD: GPU is implemented for methods `1` and `3`.
- TSP method `2` is not accepted in `main`.
- CPU methods support OpenMP threading when built with `ENABLE_OPENMP=1`.

### Common Enumeration Runs
Use a binary whose `NUM_OBJS` matches the input file. The examples below use
`NUM_OBJS=3`, `8` CPU threads, no state dominance, and a knapsack input.

Build an OpenMP CPU binary:

```bash
make clean
make NUM_OBJS=3 ENABLE_CUDA=0 ENABLE_OPENMP=1
```

Run top-down enumeration on CPU with threads (`method=1`):

```bash
./multiobj_nobjs3 data/3/knapsack/KP_p-3_n-10_ins-1.dat 1 1 0 --backend cpu --cpu-threads 8 --cpu-kernel 1
```

Run coupled enumeration on CPU with threads (`method=3`, dynamic layer cutset):

```bash
./multiobj_nobjs3 data/3/knapsack/KP_p-3_n-10_ins-1.dat 1 3 0 --backend cpu --cpu-threads 8 --cpu-kernel 3
```

Build a CUDA-enabled binary:

```bash
make clean
make NUM_OBJS=3 ENABLE_CUDA=1 ENABLE_OPENMP=1
```

Run GPU-based top-down enumeration (`method=1`):

```bash
./multiobj_nobjs3 data/3/knapsack/KP_p-3_n-10_ins-1.dat 1 1 0 --backend gpu --kernel 3
```

For set packing, keep the same method/backend pattern and change
`problem-type` to `2` with a set-packing input. For TSP, change `problem-type`
to `3`; TSP accepts CPU/GPU top-down with `method=1` and CPU/GPU coupled
enumeration with `method=3`.

## 4) Program Output
Stdout is always three lines.

For BDD problem types `1` and `2`:
- Line 1: number of Pareto solutions.
- Line 2: CPU total time, `cpu_compile_s + cpu_enumeration_s`.
- Line 3: tab-separated fields:
  - `method`
  - `state_dominance`
  - `original_width`
  - `reduced_width`
  - `original_num_nodes`
  - `reduced_num_nodes`
  - `cpu_compile_s`
  - `cpu_enumeration_s`
  - `layer_coupling`
  - `dominance_filtered_total`
  - `cpu_state_dominance_s`
  - `wall_compile_s`
  - `wall_enumeration_s`

For TSP problem type `3`:
- Line 1: number of Pareto solutions.
- Line 2: CPU total time, `cpu_compile_s + cpu_enumeration_s`.
- Line 3: `cpu_compile_s<TAB>cpu_enumeration_s<TAB>wall_compile_s<TAB>wall_enumeration_s`.

JSONL stats are written by `src/util/output_utils.cpp` and include identity, output paths, timing, memory, work counters, dominance counters, structure, metrics, and status.

## 5) Codebase Map
- `src/main.cpp`
  - CLI dispatch, instance loading, BDD/MDD construction, method/backend selection, output calls.
- `src/util/cli_parser.*`
  - Positional CLI and optional backend/output parsing.
- `src/util/output_utils.*`
  - Three-line stdout, gzip frontier CSV, JSONL stats.
- `src/util/stats.hpp`
  - `EnumerationStats` and `DDStats`.
- `src/util/omp_compat.hpp`, `src/util/cpu_affinity.*`
  - OpenMP compatibility and CPU thread pinning.
- `src/bdd/`
  - `bdd.hpp`: BDD node/arc structure and maintenance methods.
  - `bdd_alg.hpp`: BDD reduction logic; `bdd_alg.cpp` is effectively empty.
  - `pareto_frontier.hpp`: nondominated frontier container and merge/convolution logic.
  - `bdd_multiobj.hpp/.cpp`: CPU frontier algorithms, dominance filtering, MDD overloads, CUDA wrapper entry points.
  - `knapsack_bdd.*`, `indepset_bdd.*`: exact BDD constructors.
- `src/mdd/`
  - `mdd.hpp`: MDD node/arc structure.
  - `tsp_mdd.*`: exact TSP MDD constructor.
- `src/instances/`
  - Parsers for knapsack, set packing, independent set, and TSP.
  - `assignment_instance.*` is stubbed and not integrated in `main`.
- `src/cuda/`
  - CUDA top-down and coupled kernels plus stubs used when `ENABLE_CUDA=0`.
- `data/`
  - Included benchmark/input data for objective dimensions `3..7`.
- `cc/`
  - Experiment/job-script support directories.

## 6) Input Format Cheat Sheet
- Knapsack (`src/instances/knapsack_instance.cpp`):
  - `n_vars n_cons num_objs`
  - `num_objs` rows of `n_vars` objective coefficients
  - For each constraint: `n_vars` coefficients followed by one RHS.
- Set packing (`src/instances/setpacking_instance.hpp`):
  - `n_vars n_cons n_objs`
  - Objective matrix of shape `n_objs x n_vars`
  - For each constraint: `count` followed by `count` 1-based variable ids.
  - The parser converts constraint variable ids to 0-based indices.
- TSP (`src/instances/tsp_instance.cpp`):
  - `n_objs n_cities`
  - For each objective: full `n_cities x n_cities` cost matrix.
- Independent-set DIMACS parsing exists in `IndepSetInst`/`Graph`, but `main` uses it indirectly through set packing.

## 7) Algorithm Flow
BDD path for problem types `1` and `2`:
1. Read instance.
2. Build exact BDD.
3. For knapsack, reduce the BDD, update node weights, reduce again, then update weights again.
4. Compute structure stats such as width, node counts, layer sizes, and arc counts.
5. Enumerate the Pareto frontier using method `1`, `2`, or `3`.
6. Sort the frontier lexicographically before saving/printing summaries.

TSP path for problem type `3`:
1. Read TSP instance.
2. Build exact MDD using `MDDTSPConstructor`.
3. Enumerate with top-down (`method=1`) or dynamic layer cutset (`method=3`).
4. Sort the frontier lexicographically before saving/printing summaries.

State dominance:
- `state_dominance=0` disables state dominance.
- `state_dominance=1` enables available problem-specific filters.
- CPU filters exist for knapsack and set packing.
- CUDA state dominance is implemented for knapsack in the top-down BDD path.

## 8) Important Caveats
- `NOBJS` consistency matters. Rebuild when changing objective dimension.
- There is no dedicated unit-test suite in this repo.
- `data/` contains sample benchmark files, so prefer small checked-in instances for smoke tests.
- `write_frontier_gzip_csv` shells out to `gzip`; frontier saving requires `gzip` on `PATH`.
- Memory ownership is mixed:
  - BDD arc weights are often allocated with `new ObjType[NOBJS]`.
  - MDD arc weights are owned by `MDDArc` and freed in `~MDDArc`.
- If removing arcs or nodes, preserve `prev`/arc consistency and layer indices; use existing cleanup helpers where applicable.
- Do not assume CPLEX, set-covering, portfolio, or absval code exists in this branch.

## 9) Agent Workflow
- Start by reading `src/main.cpp`, `src/util/cli_parser.*`, and the file directly related to the requested change.
- Run `make -n` before full builds when changing build flags or source lists.
- For CPU-only verification, use `ENABLE_CUDA=0` to avoid requiring `nvcc`.
- For GPU changes, verify the CUDA build path and the relevant runtime branch if hardware/tooling is available.
- When adding a problem type:
  - Add or complete the parser in `src/instances/`.
  - Add a BDD or MDD constructor.
  - Extend CLI validation and usage text.
  - Extend `src/main.cpp` dispatch.
  - Decide whether state dominance and GPU support apply.
  - Update stdout/JSONL structure only deliberately, since scripts may depend on it.
