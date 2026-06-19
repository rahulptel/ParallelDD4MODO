// ----------------------------------------------------------
// CUDA Coupled (Dynamic Layer Cutset) Enumeration for MDD
// ----------------------------------------------------------

#include "enum_types.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "../bdd/bdd_multiobj.hpp"
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

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

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


struct HasBothFrontiers {
    const int* td_off;
    const int* bu_off;

    __host__ __device__ bool operator()(int node) const {
        return (td_off[node + 1] - td_off[node]) > 0 && (bu_off[node + 1] - bu_off[node]) > 0;
    }
};

struct FrontierSumScore {
    const int* td_off;
    const int* bu_off;

    __host__ __device__ long long operator()(int node) const {
        return static_cast<long long>(td_off[node + 1] - td_off[node]) +
               static_cast<long long>(bu_off[node + 1] - bu_off[node]);
    }
};

// ---------------------------------------------------------------
// Cutset coupling kernels for MDD
// ---------------------------------------------------------------


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


// Cartesian product: for each node, td[i] + bu[j] for all pairs
__global__ void materialize_cutset_products_kernel(const ObjType* td_pts,
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
__global__ void mark_globally_dominated_kernel(const ObjType* points,
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
__global__ void mark_frontier_dominated_by_batch_kernel(
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
// expand_layer_frontiers: runs expansion kernels for one MDD layer.
// Works identically for top-down and bottom-up.
// ---------------------------------------------------------------

ParetoFrontier* enumerate_mdd_coupled(MDD* mdd,
                                       EnumerationStats* stats,
                                       std::string* reason) {
    if (mdd == NULL) { set_reason(reason, "MDD is NULL"); return NULL; }
    if (mdd->num_layers <= 0) { set_reason(reason, "MDD has zero layers"); return NULL; }
    if (stats != NULL) {
        stats->std_candidates_per_layer.clear();
        stats->std_frontier_survivors_per_layer.clear();
    }

    const int num_layers = mdd->num_layers;
    clock_t t0, t1;

    // ----------------------------------------------------------
    // 1. Pack all MDD layers into flat GPU arrays
    // ----------------------------------------------------------
    const auto pack_begin = std::chrono::steady_clock::now();
    t0 = clock();
    std::vector<PackedMDDLayer> packed(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        const int num_nodes = mdd->layers[l].size();
        packed[l].num_nodes = num_nodes;

        // --- top-down: incoming arcs (for layers 1..num_layers-1) ---
        if (l > 0) {
            std::vector<int> h_off(num_nodes + 1, 0);
            for (int d = 0; d < num_nodes; ++d)
                h_off[d + 1] = h_off[d] + mdd->layers[l][d]->in_arcs_list.size();
            const int num_edges = h_off[num_nodes];
            std::vector<int> h_src(num_edges);
            std::vector<ObjType> h_wt(num_edges * NOBJS);
            int idx = 0;
            for (int d = 0; d < num_nodes; ++d) {
                for (MDDArc* a : mdd->layers[l][d]->in_arcs_list) {
                    h_src[idx] = a->tail->index;
                    for (int o = 0; o < NOBJS; ++o)
                        h_wt[idx * NOBJS + o] = a->weights[o];
                    ++idx;
                }
            }
            packed[l].td_num_edges = num_edges;
            packed[l].td_in_edge_offsets = h_off;
            packed[l].td_edge_src = h_src;
            packed[l].td_edge_weights = h_wt;
        } else {
            packed[l].td_num_edges = 0;
        }

        // --- bottom-up: outgoing arcs (for layers 0..num_layers-2) ---
        if (l < num_layers - 1) {
            std::vector<int> h_off(num_nodes + 1, 0);
            for (int d = 0; d < num_nodes; ++d)
                h_off[d + 1] = h_off[d] + mdd->layers[l][d]->out_arcs_list.size();
            const int num_edges = h_off[num_nodes];
            std::vector<int> h_src(num_edges);
            std::vector<ObjType> h_wt(num_edges * NOBJS);
            int idx = 0;
            for (int d = 0; d < num_nodes; ++d) {
                for (MDDArc* a : mdd->layers[l][d]->out_arcs_list) {
                    h_src[idx] = a->head->index;
                    for (int o = 0; o < NOBJS; ++o)
                        h_wt[idx * NOBJS + o] = a->weights[o];
                    ++idx;
                }
            }
            packed[l].bu_num_edges = num_edges;
            packed[l].bu_in_edge_offsets = h_off;
            packed[l].bu_edge_src = h_src;
            packed[l].bu_edge_weights = h_wt;
        } else {
            packed[l].bu_num_edges = 0;
        }

        // --- arc counts for heuristic ---
        std::vector<int> h_out(num_nodes), h_in(num_nodes);
        for (int d = 0; d < num_nodes; ++d) {
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
    int topdown_score = 0;
    int bottomup_score = 0;

    double total_td_time = 0.0, total_bu_time = 0.0;
    int td_iters = 0, bu_iters = 0;

    while (layer_td != layer_bu) {
        if (topdown_score <= bottomup_score) {
            // Expand top-down
            ++layer_td;
            const int num_nodes = packed[layer_td].num_nodes;
            thrust::device_vector<int> next_sizes, next_offsets;
            thrust::device_vector<ObjType> next_points;
            long long layer_candidates = 0;
            long long layer_survivors = 0;

            t0 = clock();
            if (!expand_layer_frontiers(
                    packed[layer_td].td_in_edge_offsets,
                    packed[layer_td].td_edge_src,
                    packed[layer_td].td_edge_weights,
                    packed[layer_td].td_num_edges,
                    num_nodes,
                    d_td_offsets, d_td_points,
                    next_sizes, next_offsets, next_points, reason,
                    &layer_candidates, &layer_survivors,
                    NULL, NULL,
                    NULL, NULL, NULL))
                return NULL;
            t1 = clock();
            double td_elapsed = (double)(t1-t0)/CLOCKS_PER_SEC;
            total_td_time += td_elapsed;
            ++td_iters;
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                stats->work_candidates_peak = std::max(stats->work_candidates_peak, layer_candidates);
                stats->work_frontier_survivors_total += layer_survivors;
                stats->work_frontier_peak_points = std::max(stats->work_frontier_peak_points, layer_survivors);
            }

            d_td_sizes.swap(next_sizes);
            d_td_offsets.swap(next_offsets);
            d_td_points.swap(next_points);

            // Compute layer value on GPU
            topdown_score = compute_expansion_score(d_td_offsets, packed[layer_td].out_arc_counts, num_nodes);
        } else {
            // Expand bottom-up
            --layer_bu;
            const int num_nodes = packed[layer_bu].num_nodes;
            thrust::device_vector<int> next_sizes, next_offsets;
            thrust::device_vector<ObjType> next_points;
            long long layer_candidates = 0;
            long long layer_survivors = 0;

            t0 = clock();
            if (!expand_layer_frontiers(
                    packed[layer_bu].bu_in_edge_offsets,
                    packed[layer_bu].bu_edge_src,
                    packed[layer_bu].bu_edge_weights,
                    packed[layer_bu].bu_num_edges,
                    num_nodes,
                    d_bu_offsets, d_bu_points,
                    next_sizes, next_offsets, next_points, reason,
                    &layer_candidates, &layer_survivors,
                    NULL, NULL,
                    NULL, NULL, NULL))
                return NULL;
            t1 = clock();
            double bu_elapsed = (double)(t1-t0)/CLOCKS_PER_SEC;
            total_bu_time += bu_elapsed;
            ++bu_iters;
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                stats->work_candidates_peak = std::max(stats->work_candidates_peak, layer_candidates);
                stats->work_frontier_survivors_total += layer_survivors;
                stats->work_frontier_peak_points = std::max(stats->work_frontier_peak_points, layer_survivors);
            }

            d_bu_sizes.swap(next_sizes);
            d_bu_offsets.swap(next_offsets);
            d_bu_points.swap(next_points);

            // Compute layer value on GPU (with 1.5x multiplier)
            bottomup_score = 1.5 * compute_expansion_score(d_bu_offsets, packed[layer_bu].in_arc_counts, num_nodes);
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

    long long total_products = 0;
    for (int i = 0; i < cutset_nodes; ++i) {
        int ts = h_td_off[i+1] - h_td_off[i];
        int bs = h_bu_off[i+1] - h_bu_off[i];
        long long p = (long long)ts * bs;
        total_products += p;
    }
    if (stats != NULL) {
        stats->work_join_products_total += total_products;
    }

    // Collect nonzero nodes and sort on device by CPU-like cutset priority:
    // td frontier size + bu frontier size (largest first).
    thrust::device_vector<int> d_all_nodes(cutset_nodes);
    thrust::copy(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(cutset_nodes),
        d_all_nodes.begin());

    thrust::device_vector<int> d_nz_nodes(cutset_nodes);
    HasBothFrontiers has_both{
        thrust::raw_pointer_cast(d_td_offsets.data()),
        thrust::raw_pointer_cast(d_bu_offsets.data())
    };
    auto nz_end = thrust::copy_if(
        d_all_nodes.begin(),
        d_all_nodes.end(),
        d_nz_nodes.begin(),
        has_both);
    d_nz_nodes.resize(static_cast<int>(nz_end - d_nz_nodes.begin()));

    thrust::device_vector<long long> d_nz_scores(d_nz_nodes.size());
    FrontierSumScore score_fn{
        thrust::raw_pointer_cast(d_td_offsets.data()),
        thrust::raw_pointer_cast(d_bu_offsets.data())
    };
    thrust::transform(d_nz_nodes.begin(), d_nz_nodes.end(), d_nz_scores.begin(), score_fn);
    thrust::sort_by_key(d_nz_scores.begin(), d_nz_scores.end(), d_nz_nodes.begin(), thrust::greater<long long>());

    thrust::host_vector<int> h_nz_nodes = d_nz_nodes;
    std::vector<int> nz_nodes(h_nz_nodes.begin(), h_nz_nodes.end());

    // Running frontier stored on device
    thrust::device_vector<ObjType> d_frontier_pts;
    int frontier_size = 0;

    const long long MAX_BATCH_PRODUCTS = 2000000LL;
    double time_cart = 0, time_filt = 0, time_update = 0;
    int num_batches = 0;

    int batch_begin = 0;
    while (batch_begin < (int)nz_nodes.size()) {
        // Determine batch bounds
        long long bprod_total = 0;
        int batch_end = batch_begin;
        while (batch_end < (int)nz_nodes.size()) {
            int ni = nz_nodes[batch_end];
            long long p = (long long)(h_td_off[ni+1]-h_td_off[ni]) * (h_bu_off[ni+1]-h_bu_off[ni]);
            if (bprod_total + p > MAX_BATCH_PRODUCTS && bprod_total > 0) break;
            bprod_total += p;
            ++batch_end;
        }
        if (bprod_total == 0) { batch_begin = batch_end; continue; }
        const int batch_node_count = batch_end - batch_begin;

        // Build batch metadata on host
        std::vector<int> h_btd_off(batch_node_count), h_bbu_off(batch_node_count), h_btd_sz(batch_node_count), h_bbu_sz(batch_node_count);
        std::vector<int> h_bprod_off(batch_node_count + 1, 0);
        for (int j = 0; j < batch_node_count; ++j) {
            int ni = nz_nodes[batch_begin + j];
            h_btd_off[j] = h_td_off[ni];
            h_bbu_off[j] = h_bu_off[ni];
            h_btd_sz[j]  = h_td_off[ni+1] - h_td_off[ni];
            h_bbu_sz[j]  = h_bu_off[ni+1] - h_bu_off[ni];
            h_bprod_off[j+1] = h_bprod_off[j] + h_btd_sz[j] * h_bbu_sz[j];
        }
        const int batch_product_count = h_bprod_off[batch_node_count];
        if (stats != NULL) {
            stats->work_candidates_total += batch_product_count;
        }

        // Upload batch metadata
        thrust::device_vector<int> d_btd_off(h_btd_off.begin(), h_btd_off.end());
        thrust::device_vector<int> d_bbu_off(h_bbu_off.begin(), h_bbu_off.end());
        thrust::device_vector<int> d_btd_sz(h_btd_sz.begin(), h_btd_sz.end());
        thrust::device_vector<int> d_bbu_sz(h_bbu_sz.begin(), h_bbu_sz.end());
        thrust::device_vector<int> d_bprod_off(h_bprod_off.begin(), h_bprod_off.end());

        // ---- A: Cartesian product ----
        clock_t ca = clock();
        thrust::device_vector<ObjType> d_bpts(batch_product_count * NOBJS, 0);
        materialize_cutset_products_kernel<<<batch_node_count, kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(d_td_points.data()),
            thrust::raw_pointer_cast(d_btd_off.data()),
            thrust::raw_pointer_cast(d_bu_points.data()),
            thrust::raw_pointer_cast(d_bbu_off.data()),
            thrust::raw_pointer_cast(d_btd_sz.data()),
            thrust::raw_pointer_cast(d_bbu_sz.data()),
            thrust::raw_pointer_cast(d_bprod_off.data()),
            batch_node_count,
            thrust::raw_pointer_cast(d_bpts.data()));
        if (!sync_kernel("cart_batch", reason)) return NULL;
        time_cart += (double)(clock()-ca)/CLOCKS_PER_SEC;

        // ---- B: Filter batch against frontier ----
        clock_t fa = clock();
        int batch_surv = batch_product_count;
        thrust::device_vector<ObjType> d_surv;

        if (frontier_size > 0 && batch_product_count > 0) {
            // Kill batch points dominated by any frontier point.
            thrust::device_vector<int> d_alive_f(batch_product_count, 0);
            mark_dominated_by_frontier_kernel<<<ceil_div(batch_product_count, kThreadsPerBlock), kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(d_bpts.data()), batch_product_count,
                thrust::raw_pointer_cast(d_frontier_pts.data()), frontier_size,
                thrust::raw_pointer_cast(d_alive_f.data()));
            if (!sync_kernel("filt_frontier", reason)) return NULL;

            thrust::device_vector<int> d_pfx_f(batch_product_count, 0);
            thrust::exclusive_scan(d_alive_f.begin(), d_alive_f.end(), d_pfx_f.begin());
            batch_surv = thrust::reduce(d_alive_f.begin(), d_alive_f.end(), 0);

            if (batch_surv > 0) {
                d_surv.resize(batch_surv * NOBJS);
                compact_alive_points_kernel<<<ceil_div(batch_product_count, kThreadsPerBlock), kThreadsPerBlock>>>(
                    thrust::raw_pointer_cast(d_alive_f.data()),
                    thrust::raw_pointer_cast(d_pfx_f.data()),
                    thrust::raw_pointer_cast(d_bpts.data()),
                    thrust::raw_pointer_cast(d_surv.data()), batch_product_count);
                if (!sync_kernel("scatter_filt", reason)) return NULL;
            }
        } else {
            d_surv = d_bpts;
            batch_surv = batch_product_count;
        }
        time_filt += (double)(clock()-fa)/CLOCKS_PER_SEC;

        // ---- C: Update frontier: remove dominated points, append survivors ----
        clock_t ua = clock();
        if (batch_surv > 0) {
            // Remove frontier points dominated by any batch survivor
            int kept_frontier = frontier_size;
            if (frontier_size > 0) {
                thrust::device_vector<int> d_alive_old(frontier_size, 0);
                mark_frontier_dominated_by_batch_kernel<<<ceil_div(frontier_size, kThreadsPerBlock), kThreadsPerBlock>>>(
                    thrust::raw_pointer_cast(d_frontier_pts.data()), frontier_size,
                    thrust::raw_pointer_cast(d_surv.data()), batch_surv,
                    thrust::raw_pointer_cast(d_alive_old.data()));
                if (!sync_kernel("old_dom", reason)) return NULL;

                kept_frontier = thrust::reduce(d_alive_old.begin(), d_alive_old.end(), 0);

                if (kept_frontier < frontier_size) {
                    thrust::device_vector<int> d_pfx_old(frontier_size, 0);
                    thrust::exclusive_scan(d_alive_old.begin(), d_alive_old.end(), d_pfx_old.begin());
                    thrust::device_vector<ObjType> d_kept(kept_frontier * NOBJS);
                    compact_alive_points_kernel<<<ceil_div(frontier_size, kThreadsPerBlock), kThreadsPerBlock>>>(
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
        batch_begin = batch_end;
    }
    t1 = clock();
    if (stats != NULL) {
        stats->wall_join_s += (double)(t1-t0)/CLOCKS_PER_SEC;
    }

    // ---- Final global dominance pass on accumulated frontier ----
    clock_t fg0 = clock();
    if (frontier_size > 1) {
        thrust::device_vector<int> d_alive_final(frontier_size, 0);
        mark_globally_dominated_kernel<<<ceil_div(frontier_size, kThreadsPerBlock), kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(d_frontier_pts.data()), frontier_size,
            thrust::raw_pointer_cast(d_alive_final.data()));
        if (!sync_kernel("final_dom", reason)) return NULL;

        thrust::device_vector<int> d_pfx_final(frontier_size, 0);
        thrust::exclusive_scan(d_alive_final.begin(), d_alive_final.end(), d_pfx_final.begin());
        int final_size = thrust::reduce(d_alive_final.begin(), d_alive_final.end(), 0);

        if (final_size < frontier_size) {
            thrust::device_vector<ObjType> d_final(final_size * NOBJS);
            compact_alive_points_kernel<<<ceil_div(frontier_size, kThreadsPerBlock), kThreadsPerBlock>>>(
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
