# GPU Pareto Enumeration Optimization Ideas

This note consolidates `ideas_1.md` and `ideas_2.md` against the current code.
The main issue is still the generate-all-then-filter pattern: each layer first
materializes every candidate point in global memory, then runs a per-destination
Pareto filter and scatters survivors.

## Current GPU Shape

The active GPU implementation is the v3 dynamic scheduler. The user-facing
`--kernel 3` flag is now only a compatibility option; the current dispatch does
not keep separate v1/v2 GPU variants.

Relevant code paths:

- BDD top-down GPU: `src/cuda/topdown_cuda.cu`
  - `compute_edge_counts_kernel`
  - `expand_candidates_points_kernel`
  - `compute_dst_candidate_counts_kernel`
  - `mark_dominated_by_dst_dynamic_1d_kernel`
  - `scatter_alive_points_kernel`
- MDD top-down GPU: `topdown_mdd_cuda_enumerate`, through shared
  `expand_layer_cuda`.
- MDD coupled GPU: `src/cuda/coupled_cuda.cu`
  - uses the same generate/filter/scatter expansion pattern through
    `expand_layer_cuda`
  - this checkout has materialization sites in both `topdown_cuda.cu` and
    `coupled_cuda.cu`; search both files when changing expansion kernels
  - the final cutset join already uses a fixed product batch cap
    (`MAX_BATCH_PRODUCTS`)

The memory bottleneck is `d_cand_points` / `d_cand`, allocated as
`total_candidates * NOBJS`. It reaches peak size before dominance compaction.
The code already samples GPU memory at this exact point, so the stats fields
can be reused to evaluate any change.

## Priority 1: Bounded Batched Expansion

This is the lowest-risk implementation path. It keeps the existing dominance
logic but caps the number of realized candidates per launch.

### Idea

Process a layer in batches of edges or destination nodes:

1. Compute per-edge candidate counts as today.
2. Choose a batch range whose candidate count is below a memory budget.
3. Expand only that batch into a temporary candidate buffer.
4. Run the existing per-destination dominance filter for the batch.
5. Merge the batch survivors into the destination frontiers.
6. Repeat until the layer is complete.

This is closest to the batched-expansion idea in `ideas_2.md` and the
block-based streaming idea in `ideas_1.md`.

### Fit With Current Code

The coupled cutset join already follows this pattern with
`MAX_BATCH_PRODUCTS`: generate a bounded batch, filter against the running
frontier, update the frontier, and continue. A similar design can be introduced
for layer expansion.

For the top-down BDD path, the loop around `d_cand_points` in
`topdown_cuda_enumerate` can be extracted into a helper. For the MDD paths,
`expand_layer_cuda` is the shared helper and is the best place to add the same
bounded behavior.

### Correctness Requirement

Batch-local dominance is not enough. If batches are split within the same
destination node, later batches must be filtered against survivors from earlier
batches, and earlier survivors must be removed if dominated by later batches.

The safe merge rule per destination node is:

- filter new batch candidates against current destination survivors
- remove current survivors dominated by new batch survivors
- append remaining new survivors
- optionally run a final per-destination cleanup pass

This preserves the same final frontier as the current full-layer filter.

### Suggested API

Add a memory budget parameter internally, not to the CLI initially:

```cpp
const long long max_candidate_points_per_batch = ...;
```

Start with a conservative fixed cap, then derive it from `cudaMemGetInfo` once
correctness is stable. The CLI can remain unchanged until benchmarking shows a
need for tuning.

### Launch-Overhead Caveat

Small batches can trade OOM failures for excessive kernel-launch overhead. Use
coarse destination-node slabs or edge ranges large enough to saturate the GPU,
and only shrink the batch when `cudaMemGetInfo` indicates pressure. CUDA streams
can overlap expansion/filtering for adjacent batches, but correctness still
requires the per-destination merge rule above.

## Priority 2: Fused Insert-and-Dominate

This is the most direct way to interleave generation and elimination without
multiple expand/filter launches.

### Idea

Each thread expands a parent label, computes the candidate objective vector, and
immediately attempts to insert it into the destination node's Pareto front. The
destination front is protected by a lightweight lock, typically an `atomicCAS`
spin-lock per destination node.

While holding the lock, the thread:

1. scans existing labels for that destination
2. discards the candidate if an existing label dominates it
3. compacts away existing labels dominated by the candidate
4. inserts the candidate if capacity remains
5. releases the lock

This removes the global unfiltered candidate array entirely. The only persistent
memory is the current per-node frontier plus whatever fixed-capacity or pooled
storage backs it.

### Fit With Current Code

This design would replace the current sequence:

- `expand_candidates_points_kernel`
- `mark_dominated_by_dst_dynamic_1d_kernel`
- `scatter_alive_points_kernel`

with one fused layer-transition kernel. The destination-node structure would
need lock storage, label counts, offsets, and either fixed-capacity label
segments or a global memory pool.

The current frontier representation uses compact flat arrays plus offsets, so
this is not a small patch. It is a separate implementation track, not a tweak to
the v3 dynamic scheduler.

### Risks

- Lock contention can serialize hot destination nodes.
- Fixed-capacity per-node buffers need overflow handling; a memory pool needs
  deterministic failure/retry behavior.
- Tombstone or compaction logic must preserve nondominated labels exactly.
- Warp divergence can be high when destination frontier sizes vary widely.

This is a good follow-up if batched expansion still spends too much time in
launch orchestration or still cannot fit the largest layers.

## Priority 3: On-the-Fly Candidate Reconstruction

This removes the candidate-point buffer entirely.

### Idea

Instead of writing every candidate to `d_cand_points`, the dominance kernel
reconstructs candidate `i` from:

- source node id from `edge_src`
- source point offset from `prev_offsets`
- edge weight from `edge_weights`
- candidate position within the edge

The same reconstruction is used for candidate `j` during the dominance scan.
Only survivors are written to `d_next_points`.

### Fit With Current Code

The current `edge_offsets` array already maps each edge to a candidate range.
That is enough to recover the edge for a candidate index, but a direct binary
search per comparison would be too expensive. A practical version should either:

- keep a compact candidate-to-edge mapping for the current destination segment,
  or
- launch by edge/destination tile so candidate reconstruction can be derived
  without global searches.

### Virtual Candidate Implementation Sketch

The concrete variant in `ideas_new.md` is useful enough to keep as an
implementation sketch. Treat the candidate array as virtual: a candidate is
identified by its global or segment-local index, then reconstructed from the
edge that generated it and the source frontier point on that edge.

Core device helper:

- Inputs: local candidate index, candidate segment begin, `edge_offsets`,
  `edge_begin`, `edge_end`, `edge_src`, `edge_weights`, `prev_offsets`,
  `prev_points`.
- Binary-search `edge_offsets[edge_begin..edge_end]` to find the edge whose
  candidate range contains the index.
- Compute:
  - `src_point_idx = global_idx - edge_offsets[e]`
  - `src_node = edge_src[e]`
  - `src_base = prev_offsets[src_node]`
  - objective `o` as
    `prev_points[(src_base + src_point_idx) * NOBJS + o] + edge_weights[e * NOBJS + o]`.

Dominance-kernel change:

- Remove the `points` / `d_cand_points` parameter from the per-destination
  dominance kernel.
- Pass `edge_src`, `edge_weights`, `prev_offsets`, and `prev_points`.
- Reconstruct candidate `i` into registers.
- For each comparison tile, reconstruct candidate `j` into shared memory and
  keep the existing dominance comparison logic.
- Prefer a destination-segment restricted search using that destination's
  `edge_begin`/`edge_end`. A global binary search per comparison is simpler but
  can be costly on large layers.

Scatter-kernel change:

- Replace the scatter kernel that reads from `d_cand_points`.
- For each alive candidate, reconstruct its objective vector and write directly
  to `d_next_points[alive_prefix[i]]`.

Host-side change:

- Keep edge candidate counts, edge offsets, destination candidate counts,
  alive flags, and alive prefix sums.
- Remove allocation and launch of the candidate point buffer:
  `d_cand_points` / `d_cand`.
- Launch the virtual dominance and virtual scatter kernels with edge/source
  frontier metadata.
- Apply this to both current materialization paths:
  - `src/cuda/topdown_cuda.cu`
  - `src/cuda/coupled_cuda.cu`

Memory impact:

- Eliminates the largest temporary allocation, `total_candidates * NOBJS`.
- Does not eliminate all candidate-scale memory: alive flags, prefix arrays,
  edge offsets, and count arrays can still be `O(total_candidates)` or
  `O(num_edges + num_nodes)`.
- If alive/prefix arrays are still too large, combine virtual candidates with
  destination-node batching.

### Tradeoff

This has the best peak memory behavior, but it can multiply global reads
because each dominance comparison reconstructs both points. It is more invasive
than bounded batching and should follow it unless OOM remains severe.

Useful optimizations if this path is taken:

- cache edge weights and source points in shared memory by tile
- use warp-cooperative scans so lanes share reconstructed objective values
- specialize small `NOBJS` with unrolled objective comparisons

## Priority 4: Shared-Memory Pre-Filtering

This is a performance optimization that can be layered on top of batching or
on-the-fly reconstruction.

### Idea

Before writing candidates to global memory, each block performs a local Pareto
filter in shared memory. Only locally nondominated candidates are emitted to the
global merge/filter stage.

### Fit With Current Code

The existing dominance kernels already tile comparison points into shared
memory:

- `mark_dominated_by_dst_dynamic_1d_kernel`
- `mark_dominated_1d_kernel`
- `mark_dominated_global_kernel`

The next step would be moving part of that local filtering earlier, near
candidate generation, so dominated points never enter the large global buffer.

### Limitation

Local filtering is only a pre-filter. A point that survives inside one tile can
still be dominated by a point from another tile. A final per-destination global
merge is still required.

## Priority 5: Warp-Cooperative Dominance

The current dominance kernels assign one candidate to one thread and then each
thread scans the comparison set. That is simple but creates heavy divergence and
repeated memory traffic.

### Idea

Use a warp to evaluate one candidate or a small group of candidates:

- lanes cooperatively load comparison points
- `__shfl_sync` broadcasts objective values across lanes
- `__any_sync` exits the warp as soon as any comparison dominates the candidate

This should reduce wasted work for candidates that are dominated early and can
improve memory coalescing inside large destination segments.

### Best Target

Start with `mark_dominated_by_dst_dynamic_1d_kernel` in the BDD top-down path.
That is the clearest bottleneck and has the fewest moving parts compared with
the coupled cutset update.

## Priority 6: Bounds Before Allocation

This is a cheap pruning layer if valid bounds can be computed.

### Idea

Maintain per-layer or global incumbent bounds in device memory. Before a
candidate is written, check whether even an optimistic continuation can avoid
being dominated. If not, skip the candidate before it touches `d_cand_points`.

The sharper version is an ideal-completion bound. For candidate
`y_next = y_current + w(e)` at destination state `v_next`, precompute or
maintain an optimistic vector `ideal_completion(v_next)` from that state to the
terminal. If `y_next + ideal_completion(v_next)` is dominated by an incumbent,
then the candidate cannot produce a final Pareto solution and can be discarded
before allocation.

Store small incumbent or bound sets in constant or read-only device memory when
they fit. Larger state-dependent ideal vectors belong in regular device arrays
indexed by node/layer.

### Fit With Current Code

Knapsack already has problem-specific state dominance metadata in the GPU path:

- `min_weight`
- `single_parent_id`
- `single_parent_arc`
- `apply_knapsack_state_dominance_cuda`

That makes knapsack the best first target. General BDD/MDD bounds should wait
until the meaning of the bound is explicit for each problem class.

## Lower Priority: Global Sort-Based Filtering

Sorting can reduce comparisons, especially for 2 or 3 objectives, but it is not
a drop-in change here.

The current MDD code explicitly notes that global sorting breaks the mapping
from `edge_offsets` and destination segments. A safe sort must preserve segment
boundaries, likely by sorting inside each destination segment or by sorting
compound keys `(dst, objective_0, objective_1, ...)`.

Use sort-based filtering only after the memory issue is handled, and benchmark
it against the current dynamic scheduler.

## Recommended Roadmap

1. Add bounded batched expansion in `expand_layer_cuda` for MDD top-down and
   coupled expansion.
2. Port the same batching helper into the BDD top-down path in
   `topdown_cuda_enumerate`.
3. Validate correctness by comparing line 1 of stdout and saved frontier counts
   against CPU runs on small `data/3` instances.
4. Benchmark peak GPU memory using existing `gpu_mem_peak_used_bytes` and
   `gpu_mem_peak_reserved_bytes` stats.
5. If launch overhead or temporary buffers remain the bottleneck, prototype the
   fused insert-and-dominate design on one path, preferably BDD top-down.
6. If memory is still too high but locking looks too invasive, prototype
   on-the-fly reconstruction for one path.
7. If memory is fixed but runtime is poor, add shared-memory pre-filtering and
   warp-cooperative dominance.

## Verification Commands

CPU smoke test:

```bash
make clean
make NUM_OBJS=3 ENABLE_CUDA=0 ENABLE_OPENMP=1
./multiobj_nobjs3 data/3/knapsack/KP_p-3_n-10_ins-1.dat 1 1 0 --backend cpu --cpu-threads 8 --cpu-kernel 1
```

CUDA build:

```bash
make clean
make NUM_OBJS=3 ENABLE_CUDA=1 ENABLE_OPENMP=1
```

GPU top-down smoke test:

```bash
./multiobj_nobjs3 data/3/knapsack/KP_p-3_n-10_ins-1.dat 1 1 0 --backend gpu --kernel 3
```

TSP coupled GPU path:

```bash
./multiobj_nobjs3 data/3/tsp/tsp-nobj3-ncities10-seed12870.dat 3 3 0 --backend gpu --kernel 3
```

On this checkout, CUDA runtime verification still depends on having an actual
CUDA-capable device available. A successful CUDA build alone is not enough to
claim CPU/GPU parity.
