# GPU Pareto Enumeration Autoresearch Task

## Background

This project computes Pareto frontiers for multiobjective optimization problems
using decision diagrams. The current target is GPU enumeration for TSP MDDs.

The current GPU top-down implementation is too memory hungry on at least one
5-objective, 15-city TSP instance:

```bash
data/5/tsp/tsp-nobj5-ncities15-seed13095.dat
```

The issue is the generate-all-then-filter pattern. For a layer, the GPU code
materializes all local candidate points in global memory, then performs
per-destination dominance checks and scatters surviving points. This exposes a
large temporary buffer before pruning. The failure mode on the target instance
is:

```text
thrust::system::detail::bad_alloc
std::bad_alloc: cudaErrorMemoryAllocation: out of memory
```

The CPU coupled enumeration avoids this GPU memory-out by controlling frontier
growth through dynamic layer cutset coupling. On the target instance, the
reference CPU coupled run with 16 threads completed with:

```text
solutions: 117171
cpu_total_s: 2548.99
cpu_compile_s: 0.496982
cpu_enumeration_s: 2548.5
wall_compile_s: 0.330581
wall_enumeration_s: 393.197
```

For more context, read:

- `AGENTS.md`
- `TSP_STATUS.md`
- `ideas.md`
- the relevant code in `src/`

If documentation conflicts with code, trust the code.

## Goal

Redesign GPU Pareto frontier enumeration so the target TSP instance:

1. Does not hit GPU memory limits.
2. Produces the correct number of Pareto solutions: `117171`.
3. Minimizes `wall_enumeration_s`.

The first milestone is simply a correct GPU run that does not memory-out. After
that, keep only changes that improve the best observed `wall_enumeration_s` or
substantially simplify the implementation without worsening correctness or
runtime.

## In-Scope Files

Read these first:

- `AGENTS.md`
- `TSP_STATUS.md`
- `ideas.md`
- `src/main.cpp`
- `src/cuda/topdown_cuda.hpp`
- `src/cuda/topdown_cuda.cu`
- `src/cuda/coupled_cuda.hpp`
- `src/cuda/coupled_cuda.cu`
- `src/bdd/bdd_multiobj.hpp`
- `src/bdd/bdd_multiobj.cpp`
- `src/mdd/mdd.hpp`
- `src/mdd/tsp_mdd.hpp`
- `src/mdd/tsp_mdd.cpp`
- `src/util/stats.hpp`
- `src/util/output_utils.cpp`

You may modify files under `src/`. Avoid changing CLI behavior, input data,
benchmark files, or output formats unless the experiment explicitly requires it.

## Build and Run

Build the 5-objective CUDA binary from source after each code change:

```bash
make clean
make NUM_OBJS=5 ENABLE_CUDA=1 ENABLE_OPENMP=1
```

Run the target GPU experiment:

```bash
./multiobj_nobjs5 ./data/5/tsp/tsp-nobj5-ncities15-seed13095.dat 3 1 0 --backend gpu --kernel 3
```

If the sandbox cannot see the GPU and reports:

```text
cudaGetDeviceCount failed: no CUDA-capable device is detected
```

rerun the command with host GPU/device permissions. This is an environment
issue, not an algorithmic result.

The program prints three lines for TSP:

```text
<num_solutions>
<cpu_total_s>
<cpu_compile_s> <cpu_enumeration_s> <wall_compile_s> <wall_enumeration_s>
```

Use `wall_enumeration_s` as the primary runtime metric. Do not use CPU total
time as the optimization metric for GPU experiments.

## Correctness Criteria

A GPU run is valid only if all of the following hold:

- The process exits with code `0`.
- The first stdout line is exactly `117171`.
- The run does not report CUDA out-of-memory or any other CUDA failure.
- The result comes from the GPU backend command above, not from CPU fallback.

If a run produces a different number of solutions, treat it as incorrect even
if it is faster.

## Experiment Loop

Repeat indefinitely until interrupted:

1. Inspect the current git state and identify the current best result from
   `results.tsv` and `TSP_STATUS.md`.
2. Choose one concrete idea. Prefer small, reviewable changes.
3. Modify only the necessary files in `src/`.
4. Build:
   ```bash
   make clean
   make NUM_OBJS=5 ENABLE_CUDA=1 ENABLE_OPENMP=1
   ```
5. Commit the experiment before running it:
   ```bash
   git add src
   git commit -m "experiment: <short description>"
   ```
6. Run the target command and redirect logs:
   ```bash
   mkdir -p reports/autoresearch
   timeout 10m ./multiobj_nobjs5 ./data/5/tsp/tsp-nobj5-ncities15-seed13095.dat 3 1 0 --backend gpu --kernel 3 \
     > reports/autoresearch/run.log 2> reports/autoresearch/run.err
   ```
7. Parse stdout/stderr. Determine:
   - `num_solutions`
   - `wall_enumeration_s`
   - status: `keep`, `discard`, or `crash`
8. Append one row to `results.tsv`.
9. Keep the commit only if the run is correct and improves the incumbent
   `wall_enumeration_s`, or if it is correct and materially simplifies the code.
10. If the run crashes, memory-outs, is incorrect, or is slower without a strong
    reason, record it and reset back to the previous kept commit.

Do not stop to ask whether to continue once the loop has started. If an idea
fails, log it, revert it, and try the next idea.

Hard timeout: spend at most 10 minutes on a single experiment run. If the
command exceeds 10 minutes, kill it, log `wall_enumeration_s=-1` with status
`crash`, and revert to the previous kept commit.

## Logging Results

Create `results.tsv` if it does not exist. It must be tab-separated, not
comma-separated:

```text
commit	wall_enumeration_s	status	description
```

Columns:

- `commit`: short 7-character git commit hash for the experiment. Use `none`
  only for setup or baseline rows that were not committed.
- `wall_enumeration_s`: primary runtime metric. Use `-1` for crashes, OOM, or
  incorrect solution counts.
- `status`: `keep`, `discard`, or `crash`.
- `description`: short text description. Do not use tabs in this field.

Example:

```text
commit	wall_enumeration_s	status	description
none	-1	crash	baseline GPU top-down OOM on seed13095
a1b2c3d	260.412	keep	batched MDD layer expansion capped by candidate count
b2c3d4e	-1	crash	on-the-fly reconstruction bug produced wrong frontier count
c3d4e5f	312.801	discard	smaller batch cap avoided OOM but slower than incumbent
```

Do not commit `results.tsv` unless the user explicitly asks.

## Initial Ideas

The ideas below are not a priority order or a required plan. They are only
starting hypotheses that came to mind. Use them if they are useful, combine
them, ignore them, or develop different hypotheses after reading the code and
experiment logs.

Possible directions:

- Bounded batched expansion in `expand_layer_cuda` for MDD top-down. Avoid
  allocating `total_candidates * NOBJS` for the full layer at once.
- Reuse the CPU coupled idea on GPU: expand from both ends and join at a cutset
  while bounding intermediate candidate products.
- Stream layer expansion by destination-node batches. Preserve exact
  per-destination dominance across batches by merging survivors correctly.
- Reconstruct candidates on the fly during dominance checks instead of storing
  all candidate points in `d_cand_points`.
- Add shared-memory pre-filtering before writing candidates to global memory.
- Add warp-cooperative dominance checks to reduce repeated reads and early-exit
  faster when candidates are dominated.

Treat each experiment as a hypothesis test: state the expected effect in the
commit message or log description, run the target benchmark, and keep only
correct improvements. See `ideas.md` for more detailed design notes, but do not
feel constrained by it.

## Guardrails

- Correctness beats speed.
- Exact enumeration is required. Do not introduce approximation.
- Do not change the TSP instance format or benchmark data.
- Do not hide crashes. Log them.
- Do not keep code that only works for seed `13095` by special-casing the file
  name or objective values.
- Keep changes understandable. A small robust improvement is better than a large
  brittle rewrite.
