// ----------------------------------------------------------
// CUDA layer expansion for MDD top-down/bottom-up passes
// ----------------------------------------------------------

#include "enum_types.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "../multiobj_enum.hpp"
#include "dominance_utils.cuh"
#include "enum_types.cuh"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform_reduce.h>

namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kWarpSize = 32;



__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct SquareToDoubleInt {
    __host__ __device__ double operator()(const int x) const {
        const double value = static_cast<double>(x);
        return value * value;
    }
};

inline double population_std_from_sums(const double sum,
                                       const double sum_sq,
                                       const long long count) {
    if (count <= 0) {
        return 0.0;
    }
    const double mean = sum / static_cast<double>(count);
    double variance = sum_sq / static_cast<double>(count) - mean * mean;
    if (variance < 0.0) {
        variance = 0.0;
    }
    return std::sqrt(variance);
}

inline double population_std_from_device_counts(const thrust::device_vector<int>& counts) {
    const long long count = static_cast<long long>(counts.size());
    if (count <= 0) {
        return 0.0;
    }
    const long long sum_ll = thrust::reduce(counts.begin(), counts.end(), 0LL, thrust::plus<long long>());
    const double sum_sq = thrust::transform_reduce(counts.begin(),
                                                   counts.end(),
                                                   SquareToDoubleInt(),
                                                   0.0,
                                                   thrust::plus<double>());
    return population_std_from_sums(static_cast<double>(sum_ll), sum_sq, count);
}


// ---------------------------------------------------------------
// Kernels for MDD layer expansion.
// ---------------------------------------------------------------

__global__ void count_edge_candidates_kernel(const int* edge_src,
                                           const int* prev_offsets,
                                           int* edge_counts,
                                           int num_edges) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    const int src = edge_src[e];
    edge_counts[e] = prev_offsets[src + 1] - prev_offsets[src];
}

__global__ void count_destination_candidates_kernel(const int* in_edge_offsets,
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

__global__ void materialize_edge_candidates_kernel(const int* edge_src,
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

// mark_local_dominated_kernel uses a strictly load balanced 1D grid.
__global__ void mark_local_dominated_kernel(const ObjType* points,
                                         const int* in_edge_offsets,
                                         const int* edge_offsets,
                                         const int* block_offsets,
                                         int next_nodes,
                                         int* alive,
                                         int* next_sizes) {
    const int block_index = blockIdx.x;
    const int dst = find_dst_node(block_index, block_offsets, next_nodes);
    if (dst >= next_nodes) return;

    const int tile_i = block_index - block_offsets[dst];

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
                if (dominates_or_tie_before(&sh[jj * NOBJS], pi, lj < li)) {
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

__global__ void compact_alive_points_kernel(const int* alive,
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
__global__ void compute_layer_score_kernel(const int* offsets, const int* arc_counts,
                                   int* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (offsets[i + 1] - offsets[i]) * arc_counts[i];
}


// Removed functors for global sorting since they break edge_offsets mapping

} // anonymous namespace

// ---------------------------------------------------------------
// expand_layer_frontiers: runs expansion kernels for one MDD layer.
// Works identically for top-down and bottom-up.
// ---------------------------------------------------------------
bool expand_layer_frontiers(
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
    long long* total_candidates_out,
    long long* total_next_out,
    double* std_candidates_out,
    double* std_survivors_out,
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
    if (std_candidates_out != NULL) {
        *std_candidates_out = 0.0;
    }
    if (std_survivors_out != NULL) {
        *std_survivors_out = 0.0;
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

    count_edge_candidates_kernel<<<ceil_div(num_edges, kThreadsPerBlock), kThreadsPerBlock>>>(
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

    thrust::device_vector<int> d_cc(next_nodes, 0);
    thrust::device_vector<int> d_blocks(next_nodes, 0);

    count_destination_candidates_kernel<<<ceil_div(next_nodes, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(in_edge_offsets.data()),
        thrust::raw_pointer_cast(d_eo.data()),
        next_nodes,
        thrust::raw_pointer_cast(d_cc.data()),
        thrust::raw_pointer_cast(d_blocks.data()));
    if (!sync_kernel("dst_counts", reason)) return false;
    if (std_candidates_out != NULL) {
        *std_candidates_out = population_std_from_device_counts(d_cc);
    }

    const long long max_candidate_points_per_batch = 20000000LL;
    if (total_cand > max_candidate_points_per_batch) {
        thrust::host_vector<int> h_in_edge_offsets = in_edge_offsets;
        thrust::host_vector<int> h_eo = d_eo;

        d_next_points.clear();
        d_next_sizes.assign(next_nodes, 0);

        int dst_begin = 0;
        while (dst_begin < next_nodes) {
            int dst_end = dst_begin + 1;
            long long batch_candidates =
                static_cast<long long>(h_eo[h_in_edge_offsets[dst_end]]) -
                static_cast<long long>(h_eo[h_in_edge_offsets[dst_begin]]);

            while (dst_end < next_nodes && batch_candidates < max_candidate_points_per_batch) {
                const long long next_candidates =
                    static_cast<long long>(h_eo[h_in_edge_offsets[dst_end + 1]]) -
                    static_cast<long long>(h_eo[h_in_edge_offsets[dst_begin]]);
                if (next_candidates > max_candidate_points_per_batch && batch_candidates > 0) {
                    break;
                }
                ++dst_end;
                batch_candidates = next_candidates;
            }

            if (batch_candidates <= 0) {
                ++dst_begin;
                continue;
            }

            const int edge_begin = h_in_edge_offsets[dst_begin];
            const int edge_end = h_in_edge_offsets[dst_end];
            const int batch_edges = edge_end - edge_begin;
            const int batch_nodes = dst_end - dst_begin;

            thrust::host_vector<int> h_batch_in_offsets(batch_nodes + 1, 0);
            for (int i = 0; i <= batch_nodes; ++i) {
                h_batch_in_offsets[i] = h_in_edge_offsets[dst_begin + i] - edge_begin;
            }
            thrust::device_vector<int> d_batch_in_offsets = h_batch_in_offsets;
            thrust::device_vector<int> d_batch_eo(batch_edges + 1, 0);

            thrust::exclusive_scan(d_ec.begin() + edge_begin,
                                   d_ec.begin() + edge_end,
                                   d_batch_eo.begin());
            const int batch_last_offset = d_batch_eo[batch_edges - 1];
            const int batch_last_count = d_ec[edge_end - 1];
            const int batch_total_cand = batch_last_offset + batch_last_count;
            d_batch_eo[batch_edges] = batch_total_cand;
            if (batch_total_cand <= 0) {
                dst_begin = dst_end;
                continue;
            }

            thrust::device_vector<ObjType> d_batch_cand(batch_total_cand * NOBJS, 0);
            materialize_edge_candidates_kernel<<<ceil_div(batch_edges, kThreadsPerBlock), kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(edge_src.data()) + edge_begin,
                thrust::raw_pointer_cast(edge_weights.data()) + edge_begin * NOBJS,
                thrust::raw_pointer_cast(d_batch_eo.data()),
                thrust::raw_pointer_cast(d_ec.data()) + edge_begin,
                thrust::raw_pointer_cast(d_prev_offsets.data()),
                thrust::raw_pointer_cast(d_prev_points.data()),
                batch_edges,
                thrust::raw_pointer_cast(d_batch_cand.data()));
            if (!sync_kernel("batch_expand_cand", reason)) return false;

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

            thrust::device_vector<int> d_batch_sizes(batch_nodes, 0);
            thrust::device_vector<int> d_batch_blocks(batch_nodes, 0);
            count_destination_candidates_kernel<<<ceil_div(batch_nodes, kThreadsPerBlock), kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(d_batch_in_offsets.data()),
                thrust::raw_pointer_cast(d_batch_eo.data()),
                batch_nodes,
                thrust::raw_pointer_cast(d_batch_sizes.data()),
                thrust::raw_pointer_cast(d_batch_blocks.data()));
            if (!sync_kernel("batch_dst_counts", reason)) return false;

            thrust::device_vector<int> d_batch_alive(batch_total_cand, 0);
            thrust::device_vector<int> d_batch_next_sizes(batch_nodes, 0);
            const int max_seg = thrust::reduce(d_batch_sizes.begin(),
                                               d_batch_sizes.end(),
                                               0,
                                               thrust::maximum<int>());
            if (max_seg > 0) {
                thrust::device_vector<int> d_batch_block_offsets(batch_nodes + 1, 0);
                thrust::exclusive_scan(d_batch_blocks.begin(),
                                       d_batch_blocks.end(),
                                       d_batch_block_offsets.begin());
                d_batch_block_offsets[batch_nodes] =
                    thrust::reduce(d_batch_blocks.begin(), d_batch_blocks.end(), 0);

                const int total_blocks = d_batch_block_offsets[batch_nodes];
                if (total_blocks > 0) {
                    mark_local_dominated_kernel<<<total_blocks, kThreadsPerBlock>>>(
                        thrust::raw_pointer_cast(d_batch_cand.data()),
                        thrust::raw_pointer_cast(d_batch_in_offsets.data()),
                        thrust::raw_pointer_cast(d_batch_eo.data()),
                        thrust::raw_pointer_cast(d_batch_block_offsets.data()),
                        batch_nodes,
                        thrust::raw_pointer_cast(d_batch_alive.data()),
                        thrust::raw_pointer_cast(d_batch_next_sizes.data()));
                    if (!sync_kernel("batch_dom_v3", reason)) return false;
                }
            }

            thrust::device_vector<int> d_batch_alive_prefix(batch_total_cand, 0);
            thrust::exclusive_scan(d_batch_alive.begin(),
                                   d_batch_alive.end(),
                                   d_batch_alive_prefix.begin());
            const int batch_total_next = thrust::reduce(d_batch_alive.begin(), d_batch_alive.end(), 0);

            thrust::device_vector<ObjType> d_batch_next_points(batch_total_next * NOBJS, 0);
            if (batch_total_next > 0) {
                compact_alive_points_kernel<<<ceil_div(batch_total_cand, kThreadsPerBlock), kThreadsPerBlock>>>(
                    thrust::raw_pointer_cast(d_batch_alive.data()),
                    thrust::raw_pointer_cast(d_batch_alive_prefix.data()),
                    thrust::raw_pointer_cast(d_batch_cand.data()),
                    thrust::raw_pointer_cast(d_batch_next_points.data()),
                    batch_total_cand);
                if (!sync_kernel("batch_scatter", reason)) return false;

                const size_t old_size = d_next_points.size();
                d_next_points.resize(old_size + d_batch_next_points.size());
                thrust::copy(d_batch_next_points.begin(),
                             d_batch_next_points.end(),
                             d_next_points.begin() + old_size);
            }

            thrust::copy(d_batch_next_sizes.begin(),
                         d_batch_next_sizes.end(),
                         d_next_sizes.begin() + dst_begin);

            dst_begin = dst_end;
        }

        const int total_next = d_next_points.size() / NOBJS;
        if (total_next_out != NULL) {
            *total_next_out = total_next;
        }
        if (std_survivors_out != NULL) {
            *std_survivors_out = population_std_from_device_counts(d_next_sizes);
        }

        thrust::exclusive_scan(d_next_sizes.begin(), d_next_sizes.end(), d_next_offsets.begin());
        d_next_offsets[next_nodes] = total_next;
        return true;
    }

    // Expand candidate points
    thrust::device_vector<ObjType> d_cand(total_cand * NOBJS, 0);
    materialize_edge_candidates_kernel<<<ceil_div(num_edges, kThreadsPerBlock), kThreadsPerBlock>>>(
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

    thrust::device_vector<int> d_alive(total_cand, 0);

    const int max_seg = thrust::reduce(d_cc.begin(), d_cc.end(), 0, thrust::maximum<int>());
    if (max_seg > 0) {
        thrust::device_vector<int> d_block_offsets(next_nodes + 1, 0);
        thrust::exclusive_scan(d_blocks.begin(), d_blocks.end(), d_block_offsets.begin());
        d_block_offsets[next_nodes] = thrust::reduce(d_blocks.begin(), d_blocks.end(), 0);

        const int total_blocks = d_block_offsets[next_nodes];
        if (total_blocks > 0) {
            mark_local_dominated_kernel<<<total_blocks, kThreadsPerBlock>>>(
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

    // Compact surviving points
    thrust::device_vector<int> d_ap(total_cand, 0);
    thrust::exclusive_scan(d_alive.begin(), d_alive.end(), d_ap.begin());
    const int total_next = thrust::reduce(d_next_sizes.begin(), d_next_sizes.end(), 0);
    if (total_next_out != NULL) {
        *total_next_out = total_next;
    }
    if (std_survivors_out != NULL) {
        *std_survivors_out = population_std_from_device_counts(d_next_sizes);
    }

    thrust::exclusive_scan(d_next_sizes.begin(), d_next_sizes.end(), d_next_offsets.begin());
    d_next_offsets[next_nodes] = total_next;

    d_next_points.resize(total_next * NOBJS);
    if (total_next > 0) {
        compact_alive_points_kernel<<<ceil_div(total_cand, kThreadsPerBlock), kThreadsPerBlock>>>(
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
int compute_expansion_score(const thrust::device_vector<int>& offsets,
                        const thrust::device_vector<int>& arc_counts,
                        int num_nodes) {
    if (num_nodes <= 0) return 0;
    thrust::device_vector<int> tmp(num_nodes, 0);
    compute_layer_score_kernel<<<ceil_div(num_nodes, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(offsets.data()),
        thrust::raw_pointer_cast(arc_counts.data()),
        thrust::raw_pointer_cast(tmp.data()),
        num_nodes);
    cudaDeviceSynchronize();
    return thrust::reduce(tmp.begin(), tmp.end(), 0);
}


// ---------------------------------------------------------------
// Layer expansion API
// ---------------------------------------------------------------

bool bottomup_expand_mdd_layer(
    const PackedMDDLayer& packed_layer,
    const thrust::device_vector<int>& d_prev_offsets,
    const thrust::device_vector<ObjType>& d_prev_points,
    thrust::device_vector<int>& d_next_sizes,
    thrust::device_vector<int>& d_next_offsets,
    thrust::device_vector<ObjType>& d_next_points,
    std::string* reason,
    long long* total_candidates_out,
    long long* total_next_out,
    double* std_candidates_out,
    double* std_survivors_out,
    long long* gpu_mem_baseline_used_bytes,
    long long* gpu_mem_peak_used_bytes,
    long long* gpu_mem_peak_reserved_bytes) {

    return expand_layer_frontiers(
        packed_layer.bu_in_edge_offsets,
        packed_layer.bu_edge_src,
        packed_layer.bu_edge_weights,
        packed_layer.bu_num_edges,
        packed_layer.num_nodes,
        d_prev_offsets,
        d_prev_points,
        d_next_sizes,
        d_next_offsets,
        d_next_points,
        reason,
        total_candidates_out,
        total_next_out,
        std_candidates_out,
        std_survivors_out,
        gpu_mem_baseline_used_bytes,
        gpu_mem_peak_used_bytes,
        gpu_mem_peak_reserved_bytes);
}

bool bottomup_expand_bdd_layer(
    const PackedBDDLayer& packed_layer,
    const thrust::device_vector<int>& d_prev_offsets,
    const thrust::device_vector<ObjType>& d_prev_points,
    thrust::device_vector<int>& d_next_sizes,
    thrust::device_vector<int>& d_next_offsets,
    thrust::device_vector<ObjType>& d_next_points,
    std::string* reason,
    long long* total_candidates_out,
    long long* total_next_out,
    double* std_candidates_out,
    double* std_survivors_out,
    long long* gpu_mem_baseline_used_bytes,
    long long* gpu_mem_peak_used_bytes,
    long long* gpu_mem_peak_reserved_bytes) {

    return expand_layer_frontiers(
        packed_layer.bu_in_edge_offsets,
        packed_layer.bu_edge_src,
        packed_layer.bu_edge_weights,
        packed_layer.bu_num_edges,
        packed_layer.num_nodes,
        d_prev_offsets,
        d_prev_points,
        d_next_sizes,
        d_next_offsets,
        d_next_points,
        reason,
        total_candidates_out,
        total_next_out,
        std_candidates_out,
        std_survivors_out,
        gpu_mem_baseline_used_bytes,
        gpu_mem_peak_used_bytes,
        gpu_mem_peak_reserved_bytes);
}
