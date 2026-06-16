# GPU Enumeration Notes

This file is a code-oriented map for improving GPU Pareto frontier enumeration.
For task policy and experiment rules, read `TASK.md`. For repo-level commands
and caveats, read `AGENTS.md`.

## Target Problem

Current autoresearch target:

```bash
./multiobj_nobjs5 ./data/5/tsp/tsp-nobj5-ncities15-seed13095.dat 3 1 0 --backend gpu --kernel 3
```

This is:

- `NUM_OBJS=5`
- TSP (`problem_type=3`)
- GPU top-down MDD enumeration (`method=1`)
- no state dominance

Known baseline:

- GPU top-down memory-outs on this instance.
- CPU coupled with 16 threads completes with `117171` solutions and
  `wall_enumeration_s=393.197`.

Correct GPU result must also produce exactly `117171` solutions.

## Entry Points

Main dispatch:

- `src/main.cpp`
  - TSP path starts at `problem_type == 3`.
  - GPU top-down calls:
    `BDDMultiObj::pareto_frontier_topdown_cuda(mdd, enumeration_stats, &cuda_reason)`.
  - GPU coupled calls:
    `BDDMultiObj::pareto_frontier_dynamic_layer_cutset_cuda(mdd, enumeration_stats, &cuda_reason)`.

Wrapper methods:

- `src/bdd/bdd_multiobj.cpp`
  - `BDDMultiObj::pareto_frontier_topdown_cuda(MDD*, ...)`
    calls `topdown_mdd_cuda_enumerate`.
  - `BDDMultiObj::pareto_frontier_dynamic_layer_cutset_cuda(MDD*, ...)`
    calls `coupled_cuda_enumerate`.

CUDA declarations:

- `src/cuda/topdown_cuda.hpp`
  - `topdown_cuda_enumerate` for BDDs.
  - `topdown_mdd_cuda_enumerate` for MDDs.
  - `expand_layer_cuda`, shared by MDD top-down and MDD coupled expansion.
- `src/cuda/coupled_cuda.hpp`
  - `coupled_cuda_enumerate` for MDD dynamic layer cutset.

CUDA implementations:

- `src/cuda/topdown_cuda.cu`
  - BDD GPU top-down implementation.
  - MDD GPU top-down implementation.
  - shared `expand_layer_cuda`.
- `src/cuda/coupled_cuda.cu`
  - MDD GPU coupled implementation.
  - uses `expand_layer_cuda` for top/bottom layer expansion.
  - has a bounded final cutset join via a fixed product batch cap.

## Core Data Layout

MDD layer connectivity is packed into `PackedMDDLayer`:

- `td_in_edge_offsets`: incoming arcs grouped by destination node.
- `td_edge_src`: source node index for each top-down edge.
- `td_edge_weights`: arc objective weights.
- `bu_in_edge_offsets`: outgoing/bottom-up grouping used by coupled expansion.
- `bu_edge_src`, `bu_edge_weights`: analogous bottom-up edge data.

Frontiers are stored as flat objective arrays:

- `d_prev_offsets[node]`: start point offset for a node in previous frontier.
- `d_prev_points`: flat point array, length `num_points * NOBJS`.
- `d_next_sizes[node]`: number of surviving points for a next-layer node.
- `d_next_offsets[node]`: start point offset for a next-layer node.
- `d_next_points`: flat survivor array for the next layer.

One Pareto point occupies `NOBJS` consecutive `ObjType` entries.

## Current MDD Expansion Shape

The shared hot path is `expand_layer_cuda` in `src/cuda/topdown_cuda.cu`.

Current structure:

1. Compute candidate counts per edge.
2. Prefix-sum edge counts to get `d_edge_offsets`.
3. Compute `total_candidates`.
4. Allocate:
   ```cpp
   thrust::device_vector<ObjType> d_cand_points(total_candidates * NOBJS, 0);
   ```
5. Launch candidate expansion.
6. Compute candidate counts per destination node.
7. Run per-destination dominance marking.
8. Prefix-sum alive flags.
9. Allocate/resize `d_next_points`.
10. Scatter alive candidates into `d_next_points`.

The memory bottleneck is step 4. The full layer's candidates are materialized
before dominance pruning. On hard layers, `total_candidates * NOBJS` can exceed
available GPU memory even though the final survivor frontier would fit.

## Relevant Kernels and Helpers

Look in `src/cuda/topdown_cuda.cu` for:

- `compute_edge_counts_kernel`
  - computes how many candidates each edge contributes.
- `expand_candidates_points_kernel`
  - materializes candidate objective vectors into `d_cand_points`.
- `compute_dst_candidate_counts_kernel`
  - maps edge candidate ranges into per-destination counts.
- `mark_dominated_by_dst_dynamic_1d_kernel`
  - dominance filter by destination segment.
- `scatter_alive_points_kernel`
  - compacts surviving candidates into the next frontier.
- `sample_gpu_memory_peak`
  - records GPU memory peaks into `EnumerationStats`.

For MDD top-down:

- `topdown_mdd_cuda_enumerate`
  - packs MDD layers.
  - initializes root frontier.
  - calls `expand_layer_cuda` for each layer.
  - copies terminal frontier back to host.

For MDD coupled:

- `coupled_cuda_enumerate` in `src/cuda/coupled_cuda.cu`
  - expands top and bottom frontiers dynamically.
  - uses `expand_layer_cuda` for layer propagation.
  - joins cutset frontiers at the end.
  - already has a bounded final product batching pattern; this is useful as a
    design reference for bounded layer expansion.

## Statistics

Stats live in `src/util/stats.hpp` and are written in
`src/util/output_utils.cpp`.

Useful fields:

- `wall_enumeration_s`: primary GPU runtime metric for autoresearch.
- `gpu_mem_peak_used_bytes`: baseline-adjusted peak GPU memory used.
- `gpu_mem_peak_reserved_bytes`: device-wide peak used memory.
- `work_candidates_total`: total candidate count processed.
- `work_candidates_peak`: peak candidate count.
- `work_frontier_survivors_total`: total survivors.
- `work_frontier_peak_points`: peak survivor frontier size.
- `work_join_products_total`: cutset join work in coupled runs.

For experiments, prefer `--save-stats --stats-out <path>` when useful, but do
not change the stdout format.

## First Implementation Target

The lowest-risk first target is bounded batched expansion inside
`expand_layer_cuda`.

High-level goal:

- avoid allocating `d_cand_points` for the whole layer
- process a bounded batch of destination nodes or edges
- merge survivors into destination frontiers
- preserve exact dominance across batches

Correct merge rule for a destination node split across batches:

1. Filter new batch candidates against current destination survivors.
2. Remove current survivors dominated by new batch survivors.
3. Append remaining new survivors.
4. Optionally run a final cleanup dominance pass for that destination.

Batch-local dominance alone is not correct if the same destination node appears
in multiple batches.

## Safer Batch Boundaries

Destination-node batching is usually easier to reason about than arbitrary edge
batching:

- `td_in_edge_offsets` already groups incoming arcs by destination.
- If each batch contains whole destination nodes, no destination frontier is
  split across batches.
- This avoids cross-batch per-destination merge complexity for the first
  prototype.

Tradeoff:

- A single destination node can still have too many candidates and OOM.
- If that happens, add intra-destination batching with the exact merge rule
  above.

## Correctness Checks

For the target instance:

- stdout line 1 must be `117171`.
- process exit code must be `0`.
- no CUDA OOM or CUDA error in stderr.

For smaller validation before the target:

```bash
make clean
make NUM_OBJS=3 ENABLE_CUDA=1 ENABLE_OPENMP=1
./multiobj_nobjs3 data/3/tsp/tsp-nobj3-ncities10-seed12870.dat 3 1 0 --backend gpu --kernel 3
./multiobj_nobjs3 data/3/tsp/tsp-nobj3-ncities10-seed12870.dat 3 1 0 --backend cpu --cpu-threads 8 --cpu-kernel 1
```

Compare solution counts. Exact frontier equality is better when practical, but
solution count is the minimum smoke check.

## Common Pitfalls

- Do not optimize by dropping candidates approximately. Enumeration must remain
  exact.
- Do not special-case one instance filename or seed.
- Do not treat a CUDA device discovery failure as an algorithmic failure.
- Do not use CPU total time as the GPU optimization metric; use
  `wall_enumeration_s`.
- Do not change the CLI or stdout format unless absolutely necessary.
- Be careful with `NOBJS`; rebuild when changing objective dimension.
- Thrust allocation failures can throw exceptions before explicit error checks.
  Keep experiment logs and classify these as crashes/OOM.

## Good First Experiments

1. Add a conservative destination-node batch cap in `expand_layer_cuda`.
2. Add internal instrumentation to report maximum batch candidate count through
   existing stats fields.
3. If destination batching still OOMs, add intra-destination streaming for only
   the largest destination segment.
4. Use the coupled cutset join batching pattern as a reference for bounded
   generate-filter-merge loops.
