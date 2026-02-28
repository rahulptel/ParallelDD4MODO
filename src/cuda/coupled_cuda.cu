// ----------------------------------------------------------
// CUDA Coupled (Dynamic Layer Cutset) Enumeration for MDD
// ----------------------------------------------------------

#include "coupled_cuda.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "../bdd/bdd_multiobj.hpp"
#include "topdown_cuda.hpp"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kWarpSize = 32;

inline bool set_reason(std::string* reason, const std::string& message) {
    if (reason != NULL) *reason = message;
    return false;
}

inline bool cuda_ok(cudaError_t err, const char* where, std::string* reason) {
    if (err == cudaSuccess) return true;
    std::string msg = std::string(where) + ": " + cudaGetErrorString(err);
    return set_reason(reason, msg);
}

inline bool sync_kernel(const char* where, std::string* reason) {
    if (!cuda_ok(cudaGetLastError(), where, reason)) return false;
    return cuda_ok(cudaDeviceSynchronize(), where, reason);
}

inline bool sample_gpu_memory_peak(std::string* reason,
                                   const long long baseline_used_bytes,
                                   long long* peak_used_bytes,
                                   long long* peak_reserved_bytes) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo", reason)) {
        return false;
    }

    const long long used_bytes = static_cast<long long>(total_bytes) - static_cast<long long>(free_bytes);
    if (peak_reserved_bytes != NULL && used_bytes > *peak_reserved_bytes) {
        *peak_reserved_bytes = used_bytes;
    }

    long long delta_used_bytes = used_bytes - baseline_used_bytes;
    if (delta_used_bytes < 0) {
        delta_used_bytes = 0;
    }
    if (peak_used_bytes != NULL && delta_used_bytes > *peak_used_bytes) {
        *peak_used_bytes = delta_used_bytes;
    }
    return true;
}

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------
// Kernels (same patterns as topdown_cuda.cu, adapted for MDD)
// ---------------------------------------------------------------

__global__ void compute_edge_counts_kernel(const int* edge_src,
                                           const int* prev_offsets,
                                           int* edge_counts,
                                           int num_edges) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    const int src = edge_src[e];
    edge_counts[e] = prev_offsets[src + 1] - prev_offsets[src];
}

__global__ void compute_dst_candidate_counts_kernel(const int* in_edge_offsets,
                                                    const int* edge_offsets,
                                                    int next_nodes,
                                                    int* dst_counts,
                                                    int* dst_blocks) {
    const int dst = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst >= next_nodes) return;
    const int eb = in_edge_offsets[dst];
    const int ee = in_edge_offsets[dst + 1];
    const int count = edge_offsets[ee] - edge_offsets[eb];
    dst_counts[dst] = count;
    if (dst_blocks) {
        dst_blocks[dst] = ceil_div(count, kThreadsPerBlock);
    }
}

__global__ void expand_candidates_points_kernel(const int* edge_src,
                                                const ObjType* edge_weights,
                                                const int* edge_offsets,
                                                const int* edge_counts,
                                                const int* prev_offsets,
                                                const ObjType* prev_points,
                                                int num_edges,
                                                ObjType* cand_points) {
    const int gt = blockIdx.x * blockDim.x + threadIdx.x;
    const int gw = gt / kWarpSize;
    const int lane = gt & (kWarpSize - 1);
    const int tw = (gridDim.x * blockDim.x) / kWarpSize;
    if (tw <= 0) return;

    for (int e = gw; e < num_edges; e += tw) {
        const int src = edge_src[e];
        const int sb = prev_offsets[src];
        const int ob = edge_offsets[e];
        const int cnt = edge_counts[e];
        const ObjType* w = edge_weights + e * NOBJS;

        for (int k = lane; k < cnt; k += kWarpSize) {
            const int si = sb + k;
            const int oi = ob + k;
            #pragma unroll
            for (int o = 0; o < NOBJS; ++o)
                cand_points[oi * NOBJS + o] = prev_points[si * NOBJS + o] + w[o];
        }
    }
}

// Binary search helper device function
__device__ int find_dst_node(int block_idx, const int* block_offsets, int next_nodes) {
    int low = 0, high = next_nodes;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (block_idx < block_offsets[mid + 1]) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}

__global__ void mark_dominated_by_dst_tiled_kernel(const ObjType* points,
                                                   const int* in_edge_offsets,
                                                   const int* edge_offsets,
                                                   int next_nodes,
                                                   int* alive,
                                                   int* next_sizes) {
    const int dst = blockIdx.x;
    if (dst >= next_nodes) {
        return;
    }
    const int tile_i = blockIdx.y;

    const int edge_begin = in_edge_offsets[dst];
    const int edge_end = in_edge_offsets[dst + 1];
    const int begin = edge_offsets[edge_begin];
    const int end = edge_offsets[edge_end];
    const int len = end - begin;
    if (len <= 0) {
        return;
    }

    const int local_i = tile_i * blockDim.x + threadIdx.x;
    const bool valid_i = (local_i < len);
    const int i = begin + local_i;

    ObjType point_i[NOBJS];
    if (valid_i) {
        #pragma unroll
        for (int o = 0; o < NOBJS; ++o) {
            point_i[o] = points[i * NOBJS + o];
        }
    }

    bool dominated = false;

    __shared__ ObjType sh_points[kThreadsPerBlock * NOBJS];
    for (int j_base = 0; j_base < len; j_base += blockDim.x) {
        const int j_local = j_base + threadIdx.x;
        if (j_local < len) {
            const int j = begin + j_local;
            #pragma unroll
            for (int o = 0; o < NOBJS; ++o) {
                sh_points[threadIdx.x * NOBJS + o] = points[j * NOBJS + o];
            }
        }
        __syncthreads();

        if (valid_i && !dominated) {
            const int remaining = len - j_base;
            const int tile_count = (remaining < blockDim.x ? remaining : blockDim.x);
            for (int jj = 0; jj < tile_count && !dominated; ++jj) {
                const int local_j = j_base + jj;
                if (local_j == local_i) {
                    continue;
                }

                bool ge_all = true;
                bool strict = false;
                #pragma unroll
                for (int o = 0; o < NOBJS; ++o) {
                    const ObjType a = sh_points[jj * NOBJS + o];
                    const ObjType b = point_i[o];
                    ge_all = ge_all && (a >= b);
                    strict = strict || (a > b);
                }
                if (ge_all && (strict || (local_j < local_i))) {
                    dominated = true;
                }
            }
        }
        __syncthreads();
    }

    const int keep = (valid_i && !dominated) ? 1 : 0;
    if (valid_i) {
        alive[i] = keep;
    }

    __shared__ int live_sh[kThreadsPerBlock];
    live_sh[threadIdx.x] = keep;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            live_sh[threadIdx.x] += live_sh[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&next_sizes[dst], live_sh[0]);
    }
}

__global__ void mark_dominated_by_dst_single_block_kernel(const ObjType* points,
                                                          const int* in_edge_offsets,
                                                          const int* edge_offsets,
                                                          int next_nodes,
                                                          int* alive,
                                                          int* next_sizes) {
    const int dst = blockIdx.x;
    if (dst >= next_nodes) {
        return;
    }

    const int edge_begin = in_edge_offsets[dst];
    const int edge_end = in_edge_offsets[dst + 1];
    const int begin = edge_offsets[edge_begin];
    const int end = edge_offsets[edge_end];
    const int len = end - begin;

    int live_local = 0;
    for (int local_i = threadIdx.x; local_i < len; local_i += blockDim.x) {
        const int i = begin + local_i;
        bool dominated = false;

        for (int local_j = 0; local_j < len && !dominated; ++local_j) {
            if (local_i == local_j) {
                continue;
            }

            const int j = begin + local_j;
            bool ge_all = true;
            bool strict = false;
            #pragma unroll
            for (int o = 0; o < NOBJS; ++o) {
                const ObjType a = points[j * NOBJS + o];
                const ObjType b = points[i * NOBJS + o];
                ge_all = ge_all && (a >= b);
                strict = strict || (a > b);
            }

            if (ge_all && (strict || (local_j < local_i))) {
                dominated = true;
            }
        }

        const int keep = dominated ? 0 : 1;
        alive[i] = keep;
        live_local += keep;
    }

    __shared__ int live_sh[kThreadsPerBlock];
    live_sh[threadIdx.x] = live_local;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            live_sh[threadIdx.x] += live_sh[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        next_sizes[dst] = live_sh[0];
    }
}

// mark_dominated_1d_kernel uses a strictly load balanced 1D grid.
__global__ void mark_dominated_1d_kernel(const ObjType* points,
                                         const int* in_edge_offsets,
                                         const int* edge_offsets,
                                         const int* block_offsets,
                                         int next_nodes,
                                         int* alive,
                                         int* next_sizes) {
    const int bidx = blockIdx.x;
    const int dst = find_dst_node(bidx, block_offsets, next_nodes);
    if (dst >= next_nodes) return;

    const int tile_i = bidx - block_offsets[dst];

    const int eb = in_edge_offsets[dst];
    const int ee = in_edge_offsets[dst + 1];
    const int begin = edge_offsets[eb];
    const int end = edge_offsets[ee];
    const int len = end - begin;

    const int li = tile_i * blockDim.x + threadIdx.x;
    const bool valid = (li < len);
    const int i = begin + li;

    ObjType pi[NOBJS];
    if (valid) {
        #pragma unroll
        for (int o = 0; o < NOBJS; ++o) pi[o] = points[i * NOBJS + o];
    }

    bool dom = false;
    __shared__ ObjType sh[kThreadsPerBlock * NOBJS];
    for (int jb = 0; jb < len; jb += blockDim.x) {
        const int jl = jb + threadIdx.x;
        if (jl < len) {
            const int j = begin + jl;
            #pragma unroll
            for (int o = 0; o < NOBJS; ++o)
                sh[threadIdx.x * NOBJS + o] = points[j * NOBJS + o];
        }
        __syncthreads();
        if (valid && !dom) {
            const int tc = min(len - jb, (int)blockDim.x);
            for (int jj = 0; jj < tc; ++jj) {
                const int lj = jb + jj;

                if (lj == li) continue;
                bool ge = true, strict = false;
                #pragma unroll
                for (int o = 0; o < NOBJS; ++o) {
                    ge = ge && (sh[jj * NOBJS + o] >= pi[o]);
                    strict = strict || (sh[jj * NOBJS + o] > pi[o]);
                }
                if (ge && (strict || (lj < li))) {
                    dom = true;
                    break;
                }
            }
        }
        __syncthreads();
    }

    const int keep = (valid && !dom) ? 1 : 0;
    if (valid) alive[i] = keep;

    __shared__ int lv[kThreadsPerBlock];
    lv[threadIdx.x] = keep;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) lv[threadIdx.x] += lv[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&next_sizes[dst], lv[0]);
}

__global__ void scatter_alive_kernel(const int* alive,
                                     const int* prefix,
                                     const ObjType* in_pts,
                                     ObjType* out_pts,
                                     int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !alive[i]) return;
    const int oi = prefix[i];
    for (int o = 0; o < NOBJS; ++o)
        out_pts[oi * NOBJS + o] = in_pts[i * NOBJS + o];
}

// Layer-value heuristic: sum_i( sizes[i] * arc_counts[i] )
__global__ void layer_value_kernel(const int* offsets, const int* arc_counts,
                                   int* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (offsets[i + 1] - offsets[i]) * arc_counts[i];
}

// Cartesian product: for each node, td[i] + bu[j] for all pairs
__global__ void cartesian_product_kernel(const ObjType* td_pts,
                                         const int* td_off,
                                         const ObjType* bu_pts,
                                         const int* bu_off,
                                         const int* td_sz,
                                         const int* bu_sz,
                                         const int* prod_off,
                                         int num_nodes,
                                         ObjType* out) {
    const int node = blockIdx.x;
    if (node >= num_nodes) return;
    const int ts = td_sz[node], bs = bu_sz[node];
    const int total = ts * bs;
    if (total <= 0) return;
    const int tbase = td_off[node], bbase = bu_off[node];
    const int obase = prod_off[node];

    for (int p = threadIdx.x; p < total; p += blockDim.x) {
        const int ti = p / bs, bi = p % bs;
        const int oi = obase + p;
        #pragma unroll
        for (int o = 0; o < NOBJS; ++o)
            out[oi * NOBJS + o] = td_pts[(tbase + ti) * NOBJS + o]
                                + bu_pts[(bbase + bi) * NOBJS + o];
    }
}

// Global dominance filter across all points (single flat array)
__global__ void mark_dominated_global_kernel(const ObjType* points,
                                             int num_points,
                                             int* alive) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = (i < num_points);

    ObjType pi[NOBJS];
    if (valid) {
        #pragma unroll
        for (int o = 0; o < NOBJS; ++o) pi[o] = points[i * NOBJS + o];
    }

    bool dom = false;
    __shared__ ObjType sh[kThreadsPerBlock * NOBJS];
    for (int jb = 0; jb < num_points; jb += blockDim.x) {
        const int jl = jb + threadIdx.x;
        if (jl < num_points) {
            #pragma unroll
            for (int o = 0; o < NOBJS; ++o)
                sh[threadIdx.x * NOBJS + o] = points[jl * NOBJS + o];
        }
        __syncthreads();
        if (valid && !dom) {
            const int tc = min(num_points - jb, (int)blockDim.x);
            for (int jj = 0; jj < tc && !dom; ++jj) {
                const int gj = jb + jj;
                if (gj == i) continue;
                bool ge = true, strict = false;
                #pragma unroll
                for (int o = 0; o < NOBJS; ++o) {
                    ge = ge && (sh[jj * NOBJS + o] >= pi[o]);
                    strict = strict || (sh[jj * NOBJS + o] > pi[o]);
                }
                if (ge && (strict || gj < i)) dom = true;
            }
        }
        __syncthreads();
    }
    if (valid) alive[i] = dom ? 0 : 1;
}

// Filter candidates against a separate frontier:
// for each candidate, check if any frontier point dominates it.
// alive[i] = 0 if frontier dominates candidate i, else 1.
__global__ void mark_dominated_by_frontier_kernel(
    const ObjType* candidates, int num_cand,
    const ObjType* frontier, int num_frontier,
    int* alive)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = (i < num_cand);

    ObjType ci[NOBJS];
    if (valid) {
        #pragma unroll
        for (int o = 0; o < NOBJS; ++o) ci[o] = candidates[i * NOBJS + o];
    }

    bool dom = false;
    __shared__ ObjType sh[kThreadsPerBlock * NOBJS];
    for (int jb = 0; jb < num_frontier; jb += blockDim.x) {
        const int jl = jb + threadIdx.x;
        if (jl < num_frontier) {
            #pragma unroll
            for (int o = 0; o < NOBJS; ++o)
                sh[threadIdx.x * NOBJS + o] = frontier[jl * NOBJS + o];
        }
        __syncthreads();
        if (valid && !dom) {
            const int tc = min(num_frontier - jb, (int)blockDim.x);
            for (int jj = 0; jj < tc && !dom; ++jj) {
                bool ge = true, strict = false;
                #pragma unroll
                for (int o = 0; o < NOBJS; ++o) {
                    ge = ge && (sh[jj * NOBJS + o] >= ci[o]);
                    strict = strict || (sh[jj * NOBJS + o] > ci[o]);
                }
                if (ge && strict) dom = true;
            }
        }
        __syncthreads();
    }
    if (valid) alive[i] = dom ? 0 : 1;
}

// Check which frontier points are dominated by any of the new candidates
// alive[i] = 0 if some candidate strictly dominates frontier[i], else 1.
__global__ void mark_frontier_dominated_kernel(
    const ObjType* frontier, int num_frontier,
    const ObjType* candidates, int num_cand,
    int* alive)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = (i < num_frontier);

    ObjType fi[NOBJS];
    if (valid) {
        #pragma unroll
        for (int o = 0; o < NOBJS; ++o) fi[o] = frontier[i * NOBJS + o];
    }

    bool dom = false;
    __shared__ ObjType sh[kThreadsPerBlock * NOBJS];
    for (int jb = 0; jb < num_cand; jb += blockDim.x) {
        const int jl = jb + threadIdx.x;
        if (jl < num_cand) {
            #pragma unroll
            for (int o = 0; o < NOBJS; ++o)
                sh[threadIdx.x * NOBJS + o] = candidates[jl * NOBJS + o];
        }
        __syncthreads();
        if (valid && !dom) {
            const int tc = min(num_cand - jb, (int)blockDim.x);
            for (int jj = 0; jj < tc && !dom; ++jj) {
                bool ge = true, strict = false;
                #pragma unroll
                for (int o = 0; o < NOBJS; ++o) {
                    ge = ge && (sh[jj * NOBJS + o] >= fi[o]);
                    strict = strict || (sh[jj * NOBJS + o] > fi[o]);
                }
                if (ge && strict) dom = true;
            }
        }
        __syncthreads();
    }
    if (valid) alive[i] = dom ? 0 : 1;
}

// Removed functors for global sorting since they break edge_offsets mapping

} // anonymous namespace

// ---------------------------------------------------------------
// expand_layer_cuda: runs expansion kernels for one MDD layer.
// Works identically for top-down and bottom-up.
// ---------------------------------------------------------------
bool expand_layer_cuda(
    const thrust::device_vector<int>& in_edge_offsets,
    const thrust::device_vector<int>& edge_src,
    const thrust::device_vector<ObjType>& edge_weights,
    int num_edges,
    int next_nodes,
    const thrust::device_vector<int>& d_prev_offsets,
    const thrust::device_vector<ObjType>& d_prev_points,
    thrust::device_vector<int>& d_next_sizes,
    thrust::device_vector<int>& d_next_offsets,
    thrust::device_vector<ObjType>& d_next_points,
    std::string* reason,
    int kernel_version,
    long long* total_candidates_out,
    long long* total_next_out,
    long long* gpu_mem_baseline_used_bytes,
    long long* gpu_mem_peak_used_bytes,
    long long* gpu_mem_peak_reserved_bytes)
{
    if (total_candidates_out != NULL) {
        *total_candidates_out = 0;
    }
    if (total_next_out != NULL) {
        *total_next_out = 0;
    }
    if (kernel_version < 1 || kernel_version > 3) {
        return set_reason(reason, "Invalid kernel_version (expected 1, 2, or 3)");
    }

    d_next_sizes.assign(next_nodes, 0);
    d_next_offsets.assign(next_nodes + 1, 0);

    if (num_edges == 0) {
        d_next_points.clear();
        return true;
    }

    // Per-edge candidate counts
    thrust::device_vector<int> d_ec(num_edges, 0);
    thrust::device_vector<int> d_eo(num_edges + 1, 0);

    compute_edge_counts_kernel<<<ceil_div(num_edges, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(edge_src.data()),
        thrust::raw_pointer_cast(d_prev_offsets.data()),
        thrust::raw_pointer_cast(d_ec.data()),
        num_edges);
    if (!sync_kernel("edge_counts", reason)) return false;

    thrust::exclusive_scan(d_ec.begin(), d_ec.end(), d_eo.begin());
    const int lo = d_eo[num_edges - 1];
    const int lc = d_ec[num_edges - 1];
    const int total_cand = lo + lc;
    if (total_candidates_out != NULL) {
        *total_candidates_out = total_cand;
    }
    d_eo[num_edges] = total_cand;

    if (total_cand == 0) {
        d_next_points.clear();
        return true;
    }

    // Expand candidate points
    thrust::device_vector<ObjType> d_cand(total_cand * NOBJS, 0);
    expand_candidates_points_kernel<<<ceil_div(num_edges, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(edge_src.data()),
        thrust::raw_pointer_cast(edge_weights.data()),
        thrust::raw_pointer_cast(d_eo.data()),
        thrust::raw_pointer_cast(d_ec.data()),
        thrust::raw_pointer_cast(d_prev_offsets.data()),
        thrust::raw_pointer_cast(d_prev_points.data()),
        num_edges,
        thrust::raw_pointer_cast(d_cand.data()));
    if (!sync_kernel("expand_cand", reason)) return false;
    // Primary peak checkpoint: candidate points are fully materialized and not yet compacted by dominance.
    if (gpu_mem_baseline_used_bytes != NULL &&
        gpu_mem_peak_used_bytes != NULL &&
        gpu_mem_peak_reserved_bytes != NULL)
    {
        if (!sample_gpu_memory_peak(reason,
                                    *gpu_mem_baseline_used_bytes,
                                    gpu_mem_peak_used_bytes,
                                    gpu_mem_peak_reserved_bytes))
        {
            return false;
        }
    }

    // Segmented sort of candidates by First Objective (Descending)
    // To do segmented sort with Thrust efficiently when we only have segment offsets 
    // (in_edge_offsets + d_eo for final offsets), we can create a segment-ID array 
    // and use thrust::sort_by_key or write a custom segmented sort.
    // However, for Pareto optimization, a simple `thrust::sort_by_key` with node IDs as primary keys
    // and Obj0 as secondary keys (descending) is very effective.
    
    // We already have destination node counts in d_cc. We also compute per-node block counts
    // used by the dynamic (version 3) scheduler.
    thrust::device_vector<int> d_cc(next_nodes, 0);
    thrust::device_vector<int> d_blocks(next_nodes, 0);

    compute_dst_candidate_counts_kernel<<<ceil_div(next_nodes, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(in_edge_offsets.data()),
        thrust::raw_pointer_cast(d_eo.data()),
        next_nodes,
        thrust::raw_pointer_cast(d_cc.data()),
        thrust::raw_pointer_cast(d_blocks.data()));
    if (!sync_kernel("dst_counts", reason)) return false;

    // We no longer sort globally. Candidates remain exactly in the order
    // defined by in_edge_offsets and block_offsets mapping.
    
    thrust::device_vector<int> d_alive(total_cand, 0);

    const int max_seg = thrust::reduce(d_cc.begin(), d_cc.end(), 0, thrust::maximum<int>());
    if (max_seg > 0) {
        if (kernel_version == 1) {
            // v1: one block per node
            mark_dominated_by_dst_single_block_kernel<<<next_nodes, kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(d_cand.data()),
                thrust::raw_pointer_cast(in_edge_offsets.data()),
                thrust::raw_pointer_cast(d_eo.data()),
                next_nodes,
                thrust::raw_pointer_cast(d_alive.data()),
                thrust::raw_pointer_cast(d_next_sizes.data()));
            if (!sync_kernel("dom_v1", reason)) return false;
        } else if (kernel_version == 2) {
            // v2: fixed number of blocks per node (dense 2D grid)
            dim3 mark_grid(next_nodes, ceil_div(max_seg, kThreadsPerBlock));
            mark_dominated_by_dst_tiled_kernel<<<mark_grid, kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(d_cand.data()),
                thrust::raw_pointer_cast(in_edge_offsets.data()),
                thrust::raw_pointer_cast(d_eo.data()),
                next_nodes,
                thrust::raw_pointer_cast(d_alive.data()),
                thrust::raw_pointer_cast(d_next_sizes.data()));
            if (!sync_kernel("dom_v2", reason)) return false;
        } else {
            // v3: dynamic number of blocks per node with 1D load balancing
            thrust::device_vector<int> d_block_offsets(next_nodes + 1, 0);
            thrust::exclusive_scan(d_blocks.begin(), d_blocks.end(), d_block_offsets.begin());
            d_block_offsets[next_nodes] = thrust::reduce(d_blocks.begin(), d_blocks.end(), 0);

            const int total_blocks = d_block_offsets[next_nodes];
            if (total_blocks > 0) {
                mark_dominated_1d_kernel<<<total_blocks, kThreadsPerBlock>>>(
                    thrust::raw_pointer_cast(d_cand.data()),
                    thrust::raw_pointer_cast(in_edge_offsets.data()),
                    thrust::raw_pointer_cast(d_eo.data()),
                    thrust::raw_pointer_cast(d_block_offsets.data()),
                    next_nodes,
                    thrust::raw_pointer_cast(d_alive.data()),
                    thrust::raw_pointer_cast(d_next_sizes.data()));
                if (!sync_kernel("dom_v3", reason)) return false;
            }
        }
    }

    // Compact surviving points
    thrust::device_vector<int> d_ap(total_cand, 0);
    thrust::exclusive_scan(d_alive.begin(), d_alive.end(), d_ap.begin());
    const int total_next = thrust::reduce(d_alive.begin(), d_alive.end(), 0);
    if (total_next_out != NULL) {
        *total_next_out = total_next;
    }

    thrust::exclusive_scan(d_next_sizes.begin(), d_next_sizes.end(), d_next_offsets.begin());
    d_next_offsets[next_nodes] = total_next;

    d_next_points.resize(total_next * NOBJS);
    if (total_next > 0) {
        scatter_alive_kernel<<<ceil_div(total_cand, kThreadsPerBlock), kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(d_alive.data()),
            thrust::raw_pointer_cast(d_ap.data()),
            thrust::raw_pointer_cast(d_cand.data()),
            thrust::raw_pointer_cast(d_next_points.data()),
            total_cand);
        if (!sync_kernel("scatter", reason)) return false;
    }
    return true;
}

// Compute layer value heuristic on GPU, return scalar to host
int compute_layer_value(const thrust::device_vector<int>& offsets,
                        const thrust::device_vector<int>& arc_counts,
                        int num_nodes) {
    if (num_nodes <= 0) return 0;
    thrust::device_vector<int> tmp(num_nodes, 0);
    layer_value_kernel<<<ceil_div(num_nodes, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(offsets.data()),
        thrust::raw_pointer_cast(arc_counts.data()),
        thrust::raw_pointer_cast(tmp.data()),
        num_nodes);
    cudaDeviceSynchronize();
    return thrust::reduce(tmp.begin(), tmp.end(), 0);
}


// ---------------------------------------------------------------
// Public API
// ---------------------------------------------------------------

bool coupled_cuda_available(std::string* reason) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess)
        return set_reason(reason, std::string("cudaGetDeviceCount: ") + cudaGetErrorString(err));
    if (count <= 0)
        return set_reason(reason, "No CUDA device found");
    return true;
}

ParetoFrontier* coupled_cuda_enumerate(MDD* mdd,
                                       EnumerationStats* stats,
                                       std::string* reason,
                                       int kernel_version) {
    if (mdd == NULL) { set_reason(reason, "MDD is NULL"); return NULL; }
    if (mdd->num_layers <= 0) { set_reason(reason, "MDD has zero layers"); return NULL; }
    if (!coupled_cuda_available(reason)) return NULL;
    if (!cuda_ok(cudaSetDevice(0), "cudaSetDevice", reason)) return NULL;

    const int num_layers = mdd->num_layers;
    clock_t t0, t1;

    // ----------------------------------------------------------
    // 1. Pack all MDD layers into flat GPU arrays
    // ----------------------------------------------------------
    const auto pack_begin = std::chrono::steady_clock::now();
    t0 = clock();
    std::vector<PackedMDDLayer> packed(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        const int nn = mdd->layers[l].size();
        packed[l].num_nodes = nn;

        // --- top-down: incoming arcs (for layers 1..num_layers-1) ---
        if (l > 0) {
            std::vector<int> h_off(nn + 1, 0);
            for (int d = 0; d < nn; ++d)
                h_off[d + 1] = h_off[d] + mdd->layers[l][d]->in_arcs_list.size();
            const int ne = h_off[nn];
            std::vector<int> h_src(ne);
            std::vector<ObjType> h_wt(ne * NOBJS);
            int idx = 0;
            for (int d = 0; d < nn; ++d) {
                for (MDDArc* a : mdd->layers[l][d]->in_arcs_list) {
                    h_src[idx] = a->tail->index;
                    for (int o = 0; o < NOBJS; ++o)
                        h_wt[idx * NOBJS + o] = a->weights[o];
                    ++idx;
                }
            }
            packed[l].td_num_edges = ne;
            packed[l].td_in_edge_offsets = h_off;
            packed[l].td_edge_src = h_src;
            packed[l].td_edge_weights = h_wt;
        } else {
            packed[l].td_num_edges = 0;
        }

        // --- bottom-up: outgoing arcs (for layers 0..num_layers-2) ---
        if (l < num_layers - 1) {
            std::vector<int> h_off(nn + 1, 0);
            for (int d = 0; d < nn; ++d)
                h_off[d + 1] = h_off[d] + mdd->layers[l][d]->out_arcs_list.size();
            const int ne = h_off[nn];
            std::vector<int> h_src(ne);
            std::vector<ObjType> h_wt(ne * NOBJS);
            int idx = 0;
            for (int d = 0; d < nn; ++d) {
                for (MDDArc* a : mdd->layers[l][d]->out_arcs_list) {
                    h_src[idx] = a->head->index;
                    for (int o = 0; o < NOBJS; ++o)
                        h_wt[idx * NOBJS + o] = a->weights[o];
                    ++idx;
                }
            }
            packed[l].bu_num_edges = ne;
            packed[l].bu_in_edge_offsets = h_off;
            packed[l].bu_edge_src = h_src;
            packed[l].bu_edge_weights = h_wt;
        } else {
            packed[l].bu_num_edges = 0;
        }

        // --- arc counts for heuristic ---
        std::vector<int> h_out(nn), h_in(nn);
        for (int d = 0; d < nn; ++d) {
            h_out[d] = mdd->layers[l][d]->out_arcs_list.size();
            h_in[d] = mdd->layers[l][d]->in_arcs_list.size();
        }
        packed[l].out_arc_counts = h_out;
        packed[l].in_arc_counts = h_in;
    }
    t1 = clock();
    if (stats != NULL) {
        stats->wall_pack_transfer_s += std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - pack_begin).count();
    }

    // ----------------------------------------------------------
    // 2. Initialize top-down and bottom-up frontiers
    // ----------------------------------------------------------
    const int root_nodes = packed[0].num_nodes;
    const int root_idx = mdd->get_root()->index;

    thrust::host_vector<int> h_td_sizes(root_nodes, 0);
    h_td_sizes[root_idx] = 1;
    thrust::device_vector<int> d_td_sizes = h_td_sizes;
    thrust::device_vector<int> d_td_offsets(root_nodes + 1, 0);
    thrust::exclusive_scan(d_td_sizes.begin(), d_td_sizes.end(), d_td_offsets.begin());
    d_td_offsets[root_nodes] = 1;
    thrust::device_vector<ObjType> d_td_points(NOBJS, 0);

    const int term_layer = num_layers - 1;
    const int term_nodes = packed[term_layer].num_nodes;
    const int term_idx = mdd->get_terminal()->index;

    thrust::host_vector<int> h_bu_sizes(term_nodes, 0);
    h_bu_sizes[term_idx] = 1;
    thrust::device_vector<int> d_bu_sizes = h_bu_sizes;
    thrust::device_vector<int> d_bu_offsets(term_nodes + 1, 0);
    thrust::exclusive_scan(d_bu_sizes.begin(), d_bu_sizes.end(), d_bu_offsets.begin());
    d_bu_offsets[term_nodes] = 1;
    thrust::device_vector<ObjType> d_bu_points(NOBJS, 0);

    // ----------------------------------------------------------
    // 3. Dynamic layer selection loop (all data stays on GPU)
    // ----------------------------------------------------------
    int layer_td = 0;
    int layer_bu = num_layers - 1;
    int val_td = 0;
    int val_bu = 0;

    double total_td_time = 0.0, total_bu_time = 0.0;
    int td_iters = 0, bu_iters = 0;

    while (layer_td != layer_bu) {
        if (val_td <= val_bu) {
            // Expand top-down
            ++layer_td;
            const int nn = packed[layer_td].num_nodes;
            thrust::device_vector<int> d_ns, d_no;
            thrust::device_vector<ObjType> d_np;
            long long layer_candidates = 0;
            long long layer_survivors = 0;

            t0 = clock();
            if (!expand_layer_cuda(
                    packed[layer_td].td_in_edge_offsets,
                    packed[layer_td].td_edge_src,
                    packed[layer_td].td_edge_weights,
                    packed[layer_td].td_num_edges,
                    nn,
                    d_td_offsets, d_td_points,
                    d_ns, d_no, d_np, reason, kernel_version,
                    &layer_candidates, &layer_survivors,
                    NULL, NULL, NULL))
                return NULL;
            t1 = clock();
            double td_elapsed = (double)(t1-t0)/CLOCKS_PER_SEC;
            total_td_time += td_elapsed;
            ++td_iters;
            if (stats != NULL) {
                stats->wall_expand_td_s += td_elapsed;
                stats->work_candidates_total += layer_candidates;
                stats->work_frontier_survivors_total += layer_survivors;
                stats->work_frontier_peak_points = std::max(stats->work_frontier_peak_points, layer_survivors);
            }

            d_td_sizes.swap(d_ns);
            d_td_offsets.swap(d_no);
            d_td_points.swap(d_np);

            // Compute layer value on GPU
            val_td = compute_layer_value(d_td_offsets, packed[layer_td].out_arc_counts, nn);
        } else {
            // Expand bottom-up
            --layer_bu;
            const int nn = packed[layer_bu].num_nodes;
            thrust::device_vector<int> d_ns, d_no;
            thrust::device_vector<ObjType> d_np;
            long long layer_candidates = 0;
            long long layer_survivors = 0;

            t0 = clock();
            if (!expand_layer_cuda(
                    packed[layer_bu].bu_in_edge_offsets,
                    packed[layer_bu].bu_edge_src,
                    packed[layer_bu].bu_edge_weights,
                    packed[layer_bu].bu_num_edges,
                    nn,
                    d_bu_offsets, d_bu_points,
                    d_ns, d_no, d_np, reason, kernel_version,
                    &layer_candidates, &layer_survivors,
                    NULL, NULL, NULL))
                return NULL;
            t1 = clock();
            double bu_elapsed = (double)(t1-t0)/CLOCKS_PER_SEC;
            total_bu_time += bu_elapsed;
            ++bu_iters;
            if (stats != NULL) {
                stats->wall_expand_bu_s += bu_elapsed;
                stats->work_candidates_total += layer_candidates;
                stats->work_frontier_survivors_total += layer_survivors;
                stats->work_frontier_peak_points = std::max(stats->work_frontier_peak_points, layer_survivors);
            }

            d_bu_sizes.swap(d_ns);
            d_bu_offsets.swap(d_no);
            d_bu_points.swap(d_np);

            // Compute layer value on GPU (with 1.5x multiplier)
            val_bu = 1.5 * compute_layer_value(d_bu_offsets, packed[layer_bu].in_arc_counts, nn);
        }
    }

    if (stats != NULL) stats->layer_coupling = layer_td;

    // ----------------------------------------------------------
    // 4. Cutset convolution: fully GPU-based batched pipeline
    // ----------------------------------------------------------
    t0 = clock();
    const int cutset_nodes = packed[layer_td].num_nodes;

    thrust::host_vector<int> h_td_off = d_td_offsets;
    thrust::host_vector<int> h_bu_off = d_bu_offsets;

    int total_td_pts = h_td_off[cutset_nodes];
    int total_bu_pts = h_bu_off[cutset_nodes];
    long long total_products = 0;
    int nonzero_nodes = 0;
    for (int i = 0; i < cutset_nodes; ++i) {
        int ts = h_td_off[i+1] - h_td_off[i];
        int bs = h_bu_off[i+1] - h_bu_off[i];
        long long p = (long long)ts * bs;
        total_products += p;
        if (p > 0) ++nonzero_nodes;
    }
    if (stats != NULL) {
        stats->work_join_products_total += total_products;
    }

    // Collect nonzero nodes sorted largest-first
    std::vector<int> nz_nodes;
    nz_nodes.reserve(nonzero_nodes);
    for (int i = 0; i < cutset_nodes; ++i) {
        if ((h_td_off[i+1]-h_td_off[i]) > 0 && (h_bu_off[i+1]-h_bu_off[i]) > 0)
            nz_nodes.push_back(i);
    }
    std::sort(nz_nodes.begin(), nz_nodes.end(), [&](int a, int b) {
        long long pa = (long long)(h_td_off[a+1]-h_td_off[a]) * (h_bu_off[a+1]-h_bu_off[a]);
        long long pb = (long long)(h_td_off[b+1]-h_td_off[b]) * (h_bu_off[b+1]-h_bu_off[b]);
        return pa > pb;
    });

    // Running frontier stored on device
    thrust::device_vector<ObjType> d_frontier_pts;
    int frontier_size = 0;

    const long long MAX_BATCH_PRODUCTS = 2000000LL;
    double time_cart = 0, time_filt = 0, time_update = 0;
    int num_batches = 0;

    int bidx = 0;
    while (bidx < (int)nz_nodes.size()) {
        // Determine batch bounds
        long long bprod_total = 0;
        int bend = bidx;
        while (bend < (int)nz_nodes.size()) {
            int ni = nz_nodes[bend];
            long long p = (long long)(h_td_off[ni+1]-h_td_off[ni]) * (h_bu_off[ni+1]-h_bu_off[ni]);
            if (bprod_total + p > MAX_BATCH_PRODUCTS && bprod_total > 0) break;
            bprod_total += p;
            ++bend;
        }
        if (bprod_total == 0) { bidx = bend; continue; }
        const int bc = bend - bidx;

        // Build batch metadata on host
        std::vector<int> h_btd_off(bc), h_bbu_off(bc), h_btd_sz(bc), h_bbu_sz(bc);
        std::vector<int> h_bprod_off(bc + 1, 0);
        for (int j = 0; j < bc; ++j) {
            int ni = nz_nodes[bidx + j];
            h_btd_off[j] = h_td_off[ni];
            h_bbu_off[j] = h_bu_off[ni];
            h_btd_sz[j]  = h_td_off[ni+1] - h_td_off[ni];
            h_bbu_sz[j]  = h_bu_off[ni+1] - h_bu_off[ni];
            h_bprod_off[j+1] = h_bprod_off[j] + h_btd_sz[j] * h_bbu_sz[j];
        }
        const int tbp = h_bprod_off[bc];
        if (stats != NULL) {
            stats->work_candidates_total += tbp;
        }

        // Upload batch metadata
        thrust::device_vector<int> d_btd_off(h_btd_off.begin(), h_btd_off.end());
        thrust::device_vector<int> d_bbu_off(h_bbu_off.begin(), h_bbu_off.end());
        thrust::device_vector<int> d_btd_sz(h_btd_sz.begin(), h_btd_sz.end());
        thrust::device_vector<int> d_bbu_sz(h_bbu_sz.begin(), h_bbu_sz.end());
        thrust::device_vector<int> d_bprod_off(h_bprod_off.begin(), h_bprod_off.end());

        // ---- A: Cartesian product ----
        clock_t ca = clock();
        thrust::device_vector<ObjType> d_bpts(tbp * NOBJS, 0);
        cartesian_product_kernel<<<bc, kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(d_td_points.data()),
            thrust::raw_pointer_cast(d_btd_off.data()),
            thrust::raw_pointer_cast(d_bu_points.data()),
            thrust::raw_pointer_cast(d_bbu_off.data()),
            thrust::raw_pointer_cast(d_btd_sz.data()),
            thrust::raw_pointer_cast(d_bbu_sz.data()),
            thrust::raw_pointer_cast(d_bprod_off.data()),
            bc,
            thrust::raw_pointer_cast(d_bpts.data()));
        if (!sync_kernel("cart_batch", reason)) return NULL;
        time_cart += (double)(clock()-ca)/CLOCKS_PER_SEC;

        // ---- B: Filter batch against frontier ----
        clock_t fa = clock();
        int batch_surv = tbp;
        thrust::device_vector<ObjType> d_surv;

        if (frontier_size > 0 && tbp > 0) {
            // Kill batch points dominated by any frontier point: O(tbp × frontier_size)
            thrust::device_vector<int> d_alive_f(tbp, 0);
            mark_dominated_by_frontier_kernel<<<ceil_div(tbp, kThreadsPerBlock), kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(d_bpts.data()), tbp,
                thrust::raw_pointer_cast(d_frontier_pts.data()), frontier_size,
                thrust::raw_pointer_cast(d_alive_f.data()));
            if (!sync_kernel("filt_frontier", reason)) return NULL;

            thrust::device_vector<int> d_pfx_f(tbp, 0);
            thrust::exclusive_scan(d_alive_f.begin(), d_alive_f.end(), d_pfx_f.begin());
            batch_surv = thrust::reduce(d_alive_f.begin(), d_alive_f.end(), 0);

            if (batch_surv > 0) {
                d_surv.resize(batch_surv * NOBJS);
                scatter_alive_kernel<<<ceil_div(tbp, kThreadsPerBlock), kThreadsPerBlock>>>(
                    thrust::raw_pointer_cast(d_alive_f.data()),
                    thrust::raw_pointer_cast(d_pfx_f.data()),
                    thrust::raw_pointer_cast(d_bpts.data()),
                    thrust::raw_pointer_cast(d_surv.data()), tbp);
                if (!sync_kernel("scatter_filt", reason)) return NULL;
            }
        } else {
            d_surv = d_bpts;
            batch_surv = tbp;
        }
        time_filt += (double)(clock()-fa)/CLOCKS_PER_SEC;

        // ---- C: Update frontier: remove dominated points, append survivors ----
        clock_t ua = clock();
        if (batch_surv > 0) {
            // Remove frontier points dominated by any batch survivor
            int kept_frontier = frontier_size;
            if (frontier_size > 0) {
                thrust::device_vector<int> d_alive_old(frontier_size, 0);
                mark_frontier_dominated_kernel<<<ceil_div(frontier_size, kThreadsPerBlock), kThreadsPerBlock>>>(
                    thrust::raw_pointer_cast(d_frontier_pts.data()), frontier_size,
                    thrust::raw_pointer_cast(d_surv.data()), batch_surv,
                    thrust::raw_pointer_cast(d_alive_old.data()));
                if (!sync_kernel("old_dom", reason)) return NULL;

                kept_frontier = thrust::reduce(d_alive_old.begin(), d_alive_old.end(), 0);

                if (kept_frontier < frontier_size) {
                    thrust::device_vector<int> d_pfx_old(frontier_size, 0);
                    thrust::exclusive_scan(d_alive_old.begin(), d_alive_old.end(), d_pfx_old.begin());
                    thrust::device_vector<ObjType> d_kept(kept_frontier * NOBJS);
                    scatter_alive_kernel<<<ceil_div(frontier_size, kThreadsPerBlock), kThreadsPerBlock>>>(
                        thrust::raw_pointer_cast(d_alive_old.data()),
                        thrust::raw_pointer_cast(d_pfx_old.data()),
                        thrust::raw_pointer_cast(d_frontier_pts.data()),
                        thrust::raw_pointer_cast(d_kept.data()), frontier_size);
                    if (!sync_kernel("scatter_kept", reason)) return NULL;
                    d_frontier_pts.swap(d_kept);
                }
            }

            // Append batch survivors to frontier
            int new_total = kept_frontier + batch_surv;
            d_frontier_pts.resize(new_total * NOBJS);
            thrust::copy_n(d_surv.begin(), batch_surv * NOBJS,
                           d_frontier_pts.begin() + kept_frontier * NOBJS);
            frontier_size = new_total;
        }
        time_update += (double)(clock()-ua)/CLOCKS_PER_SEC;
        if (stats != NULL) {
            stats->work_frontier_survivors_total += batch_surv;
            stats->work_frontier_peak_points = std::max(stats->work_frontier_peak_points, static_cast<long long>(frontier_size));
        }

        ++num_batches;
        bidx = bend;
    }
    t1 = clock();
    if (stats != NULL) {
        stats->wall_join_s += (double)(t1-t0)/CLOCKS_PER_SEC;
    }

    // ---- Final global dominance pass on accumulated frontier ----
    clock_t fg0 = clock();
    if (frontier_size > 1) {
        thrust::device_vector<int> d_alive_final(frontier_size, 0);
        mark_dominated_global_kernel<<<ceil_div(frontier_size, kThreadsPerBlock), kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(d_frontier_pts.data()), frontier_size,
            thrust::raw_pointer_cast(d_alive_final.data()));
        if (!sync_kernel("final_dom", reason)) return NULL;

        thrust::device_vector<int> d_pfx_final(frontier_size, 0);
        thrust::exclusive_scan(d_alive_final.begin(), d_alive_final.end(), d_pfx_final.begin());
        int final_size = thrust::reduce(d_alive_final.begin(), d_alive_final.end(), 0);

        if (final_size < frontier_size) {
            thrust::device_vector<ObjType> d_final(final_size * NOBJS);
            scatter_alive_kernel<<<ceil_div(frontier_size, kThreadsPerBlock), kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(d_alive_final.data()),
                thrust::raw_pointer_cast(d_pfx_final.data()),
                thrust::raw_pointer_cast(d_frontier_pts.data()),
                thrust::raw_pointer_cast(d_final.data()), frontier_size);
            if (!sync_kernel("scatter_final", reason)) return NULL;
            d_frontier_pts.swap(d_final);
            frontier_size = final_size;
        }
    }
    clock_t fg1 = clock();
    if (stats != NULL) {
        stats->wall_join_s += (double)(fg1-fg0)/CLOCKS_PER_SEC;
        stats->work_frontier_peak_points = std::max(stats->work_frontier_peak_points, static_cast<long long>(frontier_size));
    }

    // Copy final frontier to host
    ParetoFrontier* frontier = new ParetoFrontier;
    frontier->sols.resize(frontier_size * NOBJS, 0);
    if (frontier_size > 0) {
        thrust::host_vector<ObjType> h_res(d_frontier_pts.begin(),
                                           d_frontier_pts.begin() + frontier_size * NOBJS);
        std::copy(h_res.begin(), h_res.end(), frontier->sols.begin());
    }

    if (reason) reason->clear();
    return frontier;
}
