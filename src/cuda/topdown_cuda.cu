// ----------------------------------------------------------
// CUDA Top-Down Enumeration for BDD - Implementation
// ----------------------------------------------------------

#include "topdown_cuda.hpp"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "../bdd/bdd_multiobj.hpp"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kWarpSize = 32;

struct LayerDominanceContext {
    thrust::device_vector<int>* next_sizes;
    thrust::device_vector<int>* next_offsets;
    thrust::device_vector<ObjType>* next_points;
    thrust::device_vector<int>* layer_min_weight;
    thrust::device_vector<int>* layer_single_parent_id;
    thrust::device_vector<int>* layer_single_parent_arc;
    std::string* reason;
    bool* warned_non_knapsack;
    bool failed;
};

thread_local LayerDominanceContext* g_layer_dom_ctx = NULL;

struct PackedCudaLayer {
    thrust::device_vector<int> in_edge_offsets;
    thrust::device_vector<int> edge_src;
    thrust::device_vector<ObjType> edge_weights;
    thrust::device_vector<int> min_weight;
    thrust::device_vector<int> single_parent_id;
    thrust::device_vector<int> single_parent_arc;
};

inline bool set_reason(std::string* reason, const std::string& message) {
    if (reason != NULL) {
        *reason = message;
    }
    return false;
}

inline bool cuda_ok(cudaError_t err, const char* where, std::string* reason) {
    if (err == cudaSuccess) {
        return true;
    }
    std::string msg = where;
    msg += ": ";
    msg += cudaGetErrorString(err);
    return set_reason(reason, msg);
}

inline bool sync_kernel(const char* where, std::string* reason) {
    if (!cuda_ok(cudaGetLastError(), where, reason)) {
        return false;
    }
    return cuda_ok(cudaDeviceSynchronize(), where, reason);
}

inline bool fail_layer_filter(LayerDominanceContext* ctx, const std::string& message) {
    if (ctx != NULL) {
        ctx->failed = true;
    }
    return set_reason(ctx != NULL ? ctx->reason : NULL, message);
}

__global__ void compute_edge_counts_kernel(const int* edge_src,
                                           const int* prev_offsets,
                                           int* edge_counts,
                                           int num_edges) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) {
        return;
    }
    const int src = edge_src[e];
    edge_counts[e] = prev_offsets[src + 1] - prev_offsets[src];
}

__global__ void compute_dst_candidate_counts_kernel(const int* in_edge_offsets,
                                                    const int* edge_offsets,
                                                    int next_nodes,
                                                    int* dst_counts) {
    const int dst = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst >= next_nodes) {
        return;
    }
    const int edge_begin = in_edge_offsets[dst];
    const int edge_end = in_edge_offsets[dst + 1];
    dst_counts[dst] = edge_offsets[edge_end] - edge_offsets[edge_begin];
}

__global__ void expand_candidates_points_kernel(const int* edge_src,
                                                const ObjType* edge_weights,
                                                const int* edge_offsets,
                                                const int* edge_counts,
                                                const int* prev_offsets,
                                                const ObjType* prev_points,
                                                int num_edges,
                                                ObjType* cand_points) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp = global_thread / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int total_warps = (gridDim.x * blockDim.x) / kWarpSize;
    if (total_warps <= 0) {
        return;
    }

    for (int e = global_warp; e < num_edges; e += total_warps) {
        const int src = edge_src[e];
        const int src_begin = prev_offsets[src];
        const int out_begin = edge_offsets[e];
        const int count = edge_counts[e];
        const ObjType* w = edge_weights + (e * NOBJS);

        for (int k = lane; k < count; k += kWarpSize) {
            const int src_idx = src_begin + k;
            const int out_idx = out_begin + k;

            #pragma unroll
            for (int o = 0; o < NOBJS; ++o) {
                cand_points[out_idx * NOBJS + o] = prev_points[src_idx * NOBJS + o] + w[o];
            }
        }
    }
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

__global__ void scatter_alive_points_kernel(const int* alive,
                                            const int* alive_prefix,
                                            const ObjType* in_points,
                                            ObjType* out_points,
                                            int num_points) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points || !alive[i]) {
        return;
    }

    const int out_idx = alive_prefix[i];
    for (int o = 0; o < NOBJS; ++o) {
        out_points[out_idx * NOBJS + o] = in_points[i * NOBJS + o];
    }
}

__global__ void mark_dominated_knapsack_pairs_kernel(const int* target_nodes,
                                                     const int* cmp_node_a,
                                                     const int* cmp_node_b,
                                                     const int* offsets,
                                                     const ObjType* points,
                                                     int num_targets,
                                                     int* keep) {
    const int t = blockIdx.x;
    if (t >= num_targets) {
        return;
    }

    const int target_node = target_nodes[t];
    const int cmpA = cmp_node_a[t];
    const int cmpB = cmp_node_b[t];

    const int target_begin = offsets[target_node];
    const int target_end = offsets[target_node + 1];

    for (int local_i = threadIdx.x; local_i < target_end - target_begin; local_i += blockDim.x) {
        const int target_idx = target_begin + local_i;
        bool dominated = false;

        const int cmp_nodes[2] = {cmpA, cmpB};
        for (int c = 0; c < 2 && !dominated; ++c) {
            const int cmp_node = cmp_nodes[c];
            if (cmp_node < 0) {
                continue;
            }

            const int cmp_begin = offsets[cmp_node];
            const int cmp_end = offsets[cmp_node + 1];
            for (int cmp_idx = cmp_begin; cmp_idx < cmp_end && !dominated; ++cmp_idx) {
                dominated = true;
                for (int o = 0; o < NOBJS && dominated; ++o) {
                    const ObjType a = points[cmp_idx * NOBJS + o];
                    const ObjType b = points[target_idx * NOBJS + o];
                    dominated = (a >= b);
                }
            }
        }

        keep[target_idx] = dominated ? 0 : 1;
    }
}

__global__ void recompute_sizes_from_keep_kernel(const int* offsets,
                                                 const int* keep,
                                                 int num_nodes,
                                                 int* out_sizes) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) {
        return;
    }

    const int begin = offsets[node];
    const int end = offsets[node + 1];
    int live = 0;
    for (int i = begin; i < end; ++i) {
        live += keep[i];
    }
    out_sizes[node] = live;
}

__global__ void build_knapsack_cmp_pairs_kernel(const int* active_nodes,
                                                int active_size,
                                                const int* single_parent_id,
                                                const int* single_parent_arc,
                                                int* target_nodes,
                                                int* cmp_node_a,
                                                int* cmp_node_b) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= active_size - 1) {
        return;
    }

    const int node1 = active_nodes[i];
    const int p1 = single_parent_id[node1];
    const int a1 = single_parent_arc[node1];

    int cmpA = -1;
    int cmpB = -1;
    int cmp_count = 0;

    for (int delta = 1; delta <= 2 && i + delta < active_size; ++delta) {
        const int node2 = active_nodes[i + delta];
        const int p2 = single_parent_id[node2];
        const int a2 = single_parent_arc[node2];
        const bool skip = (p1 >= 0 && p2 >= 0 && p1 == p2 && a1 != a2);
        if (skip) {
            continue;
        }
        if (cmp_count == 0) {
            cmpA = node2;
        } else if (cmp_count == 1) {
            cmpB = node2;
        }
        ++cmp_count;
        if (cmp_count >= 2) {
            break;
        }
    }

    target_nodes[i] = node1;
    cmp_node_a[i] = cmpA;
    cmp_node_b[i] = cmpB;
}

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

struct IsPositiveInt {
    __host__ __device__ bool operator()(int x) const { return x > 0; }
};

inline bool apply_knapsack_state_dominance_cuda(BDD* bdd,
                                                const int layer,
                                                MultiObjectiveStats* stats,
                                                LayerDominanceContext* ctx) {
    if (ctx == NULL || ctx->next_sizes == NULL || ctx->next_offsets == NULL || ctx->next_points == NULL) {
        return fail_layer_filter(ctx, "Invalid CUDA dominance layer context");
    }
    if (ctx->layer_min_weight == NULL || ctx->layer_single_parent_id == NULL || ctx->layer_single_parent_arc == NULL) {
        return fail_layer_filter(ctx, "CUDA knapsack dominance filtering requires packed per-layer metadata");
    }

    const int next_nodes = ctx->next_sizes->size();
    if (next_nodes <= 1) {
        return true;
    }
    if (ctx->next_sizes->size() != static_cast<size_t>(next_nodes) ||
        ctx->next_offsets->size() != static_cast<size_t>(next_nodes + 1))
    {
        return fail_layer_filter(ctx, "CUDA dominance vectors do not match current layer size");
    }
    if (ctx->layer_min_weight->size() != static_cast<size_t>(next_nodes) ||
        ctx->layer_single_parent_id->size() != static_cast<size_t>(next_nodes) ||
        ctx->layer_single_parent_arc->size() != static_cast<size_t>(next_nodes))
    {
        return fail_layer_filter(ctx, "CUDA dominance metadata vectors do not match current layer size");
    }

    thrust::device_vector<int> d_active_nodes(next_nodes, 0);
    auto active_end = thrust::copy_if(thrust::make_counting_iterator<int>(0),
                                      thrust::make_counting_iterator<int>(next_nodes),
                                      ctx->next_sizes->begin(),
                                      d_active_nodes.begin(),
                                      IsPositiveInt());
    const int active_size = active_end - d_active_nodes.begin();
    if (active_size <= 1) {
        return true;
    }
    d_active_nodes.resize(active_size);

    thrust::device_vector<int> d_keys(active_size, 0);
    thrust::gather(d_active_nodes.begin(), d_active_nodes.end(), ctx->layer_min_weight->begin(), d_keys.begin());
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_active_nodes.begin(), thrust::greater<int>());

    const int num_targets = active_size - 1;
    thrust::device_vector<int> d_target_nodes(num_targets, 0);
    thrust::device_vector<int> d_cmp_node_a(num_targets, -1);
    thrust::device_vector<int> d_cmp_node_b(num_targets, -1);

    build_knapsack_cmp_pairs_kernel<<<ceil_div(num_targets, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(d_active_nodes.data()),
        active_size,
        thrust::raw_pointer_cast(ctx->layer_single_parent_id->data()),
        thrust::raw_pointer_cast(ctx->layer_single_parent_arc->data()),
        thrust::raw_pointer_cast(d_target_nodes.data()),
        thrust::raw_pointer_cast(d_cmp_node_a.data()),
        thrust::raw_pointer_cast(d_cmp_node_b.data()));
    if (!sync_kernel("build_knapsack_cmp_pairs_kernel", ctx->reason)) {
        return false;
    }

    const int old_total = ctx->next_points->size() / NOBJS;
    if (old_total <= 0) {
        return true;
    }
    const int last_offset = (*(ctx->next_offsets))[next_nodes];
    if (last_offset != old_total) {
        return fail_layer_filter(ctx, "CUDA dominance offsets do not match points count");
    }

    thrust::device_vector<int> d_keep(old_total, 1);
    mark_dominated_knapsack_pairs_kernel<<<num_targets, kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(d_target_nodes.data()),
        thrust::raw_pointer_cast(d_cmp_node_a.data()),
        thrust::raw_pointer_cast(d_cmp_node_b.data()),
        thrust::raw_pointer_cast(ctx->next_offsets->data()),
        thrust::raw_pointer_cast(ctx->next_points->data()),
        num_targets,
        thrust::raw_pointer_cast(d_keep.data()));
    if (!sync_kernel("mark_dominated_knapsack_pairs_kernel", ctx->reason)) {
        return false;
    }

    thrust::device_vector<int> d_filtered_sizes(next_nodes, 0);
    recompute_sizes_from_keep_kernel<<<ceil_div(next_nodes, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(ctx->next_offsets->data()),
        thrust::raw_pointer_cast(d_keep.data()),
        next_nodes,
        thrust::raw_pointer_cast(d_filtered_sizes.data()));
    if (!sync_kernel("recompute_sizes_from_keep_kernel", ctx->reason)) {
        return false;
    }

    thrust::device_vector<int> d_filtered_offsets(next_nodes + 1, 0);
    thrust::exclusive_scan(d_filtered_sizes.begin(), d_filtered_sizes.end(), d_filtered_offsets.begin());
    const int new_total = thrust::reduce(d_keep.begin(), d_keep.end(), 0);
    d_filtered_offsets[next_nodes] = new_total;

    thrust::device_vector<ObjType> d_filtered_points(new_total * NOBJS, 0);
    if (new_total > 0) {
        thrust::device_vector<int> d_keep_prefix(old_total, 0);
        thrust::exclusive_scan(d_keep.begin(), d_keep.end(), d_keep_prefix.begin());

        scatter_alive_points_kernel<<<ceil_div(old_total, kThreadsPerBlock), kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(d_keep.data()),
            thrust::raw_pointer_cast(d_keep_prefix.data()),
            thrust::raw_pointer_cast(ctx->next_points->data()),
            thrust::raw_pointer_cast(d_filtered_points.data()),
            old_total);
        if (!sync_kernel("scatter_alive_points_kernel_dominance", ctx->reason)) {
            return false;
        }
    }

    ctx->next_sizes->swap(d_filtered_sizes);
    ctx->next_offsets->swap(d_filtered_offsets);
    ctx->next_points->swap(d_filtered_points);

    if (stats != NULL && old_total > new_total) {
        stats->pareto_dominance_filtered += old_total - new_total;
    }
    return true;
}

} // namespace

void BDDMultiObj::filter_dominance_cuda(BDD* bdd,
                                        const int layer,
                                        const int problem_type,
                                        const int dominance_strategy,
                                        MultiObjectiveStats* stats) {
    if (dominance_strategy <= 0) {
        return;
    }
    if (problem_type == 1) {
        filter_dominance_knapsack_cuda(bdd, layer, stats);
        return;
    }

    LayerDominanceContext* ctx = g_layer_dom_ctx;
    if (ctx != NULL && ctx->warned_non_knapsack != NULL && !*(ctx->warned_non_knapsack)) {
        cout << "Warning: CUDA state-dominance filtering is only implemented for knapsack (problem_type=1); skipping dominance filter." << endl;
        *(ctx->warned_non_knapsack) = true;
    }
}

void BDDMultiObj::filter_dominance_knapsack_cuda(BDD* bdd, const int layer, MultiObjectiveStats* stats) {
    LayerDominanceContext* ctx = g_layer_dom_ctx;
    if (ctx == NULL) {
        return;
    }
    if (!apply_knapsack_state_dominance_cuda(bdd, layer, stats, ctx)) {
        ctx->failed = true;
    }
}


bool topdown_cuda_available(std::string* reason) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        return set_reason(reason, std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(err));
    }
    if (device_count <= 0) {
        return set_reason(reason, "No CUDA device found");
    }
    return true;
}


ParetoFrontier* topdown_cuda_enumerate(BDD* bdd,
                                       bool maximization,
                                       const int problem_type,
                                       const int dominance_strategy,
                                       MultiObjectiveStats* stats,
                                       std::string* reason) {
    if (bdd == NULL) {
        set_reason(reason, "BDD pointer is NULL");
        return NULL;
    }
    if (bdd->num_layers <= 0) {
        set_reason(reason, "BDD has zero layers");
        return NULL;
    }
    if (!topdown_cuda_available(reason)) {
        return NULL;
    }
    if (!cuda_ok(cudaSetDevice(0), "cudaSetDevice", reason)) {
        return NULL;
    }

    const int root_idx = bdd->get_root()->index;
    const int root_nodes = bdd->layers[0].size();
    if (root_idx < 0 || root_idx >= root_nodes) {
        set_reason(reason, "Invalid root index for layer 0");
        return NULL;
    }

    std::vector<PackedCudaLayer> packed_layers;
    packed_layers.resize(bdd->num_layers);

    const int first_arc_type = maximization ? 1 : 0;
    const int second_arc_type = maximization ? 0 : 1;
    const bool pack_knapsack_meta = (problem_type == 1 && dominance_strategy > 0);

    for (int l = 1; l < bdd->num_layers; ++l) {
        const int next_nodes = bdd->layers[l].size();
        std::vector<int> h_in_offsets(next_nodes + 1, 0);

        for (int dst_idx = 0; dst_idx < next_nodes; ++dst_idx) {
            Node* dst_node = bdd->layers[l][dst_idx];
            int count = 0;
            const int arc_order[2] = {first_arc_type, second_arc_type};
            for (int arc_pos = 0; arc_pos < 2; ++arc_pos) {
                count += dst_node->prev[arc_order[arc_pos]].size();
            }
            h_in_offsets[dst_idx + 1] = h_in_offsets[dst_idx] + count;
        }

        const int num_edges = h_in_offsets[next_nodes];
        std::vector<int> h_edge_src;
        std::vector<ObjType> h_edge_weights;
        h_edge_src.reserve(num_edges);
        h_edge_weights.reserve(static_cast<size_t>(num_edges) * NOBJS);

        std::vector<int> h_min_weight;
        std::vector<int> h_parent_id;
        std::vector<int> h_parent_arc;
        if (pack_knapsack_meta) {
            h_min_weight.resize(next_nodes);
            h_parent_id.resize(next_nodes);
            h_parent_arc.resize(next_nodes);
        }

        const int arc_order[2] = {first_arc_type, second_arc_type};
        for (int dst_idx = 0; dst_idx < next_nodes; ++dst_idx) {
            Node* dst_node = bdd->layers[l][dst_idx];
            for (int arc_pos = 0; arc_pos < 2; ++arc_pos) {
                const int arc_type = arc_order[arc_pos];
                for (std::vector<Node*>::iterator it = dst_node->prev[arc_type].begin();
                     it != dst_node->prev[arc_type].end(); ++it) {
                    Node* src_node = *it;
                    h_edge_src.push_back(src_node->index);

                    ObjType* w = src_node->weights[arc_type];
                    for (int o = 0; o < NOBJS; ++o) {
                        h_edge_weights.push_back(w != NULL ? w[o] : 0);
                    }
                }
            }

            if (pack_knapsack_meta) {
                h_min_weight[dst_idx] = dst_node->min_weight;
                const int parents_total = dst_node->prev[0].size() + dst_node->prev[1].size();
                if (parents_total == 1) {
                    if (dst_node->prev[0].size() == 1) {
                        h_parent_id[dst_idx] = dst_node->prev[0][0]->index;
                        h_parent_arc[dst_idx] = 0;
                    } else {
                        h_parent_id[dst_idx] = dst_node->prev[1][0]->index;
                        h_parent_arc[dst_idx] = 1;
                    }
                } else {
                    h_parent_id[dst_idx] = -1;
                    h_parent_arc[dst_idx] = -1;
                }
            }
        }

        if (static_cast<int>(h_edge_src.size()) != num_edges ||
            static_cast<int>(h_edge_weights.size()) != num_edges * NOBJS) {
            set_reason(reason, "Internal error packing BDD edges for CUDA enumeration");
            return NULL;
        }

        packed_layers[l].in_edge_offsets = h_in_offsets;
        packed_layers[l].edge_src = h_edge_src;
        packed_layers[l].edge_weights = h_edge_weights;
        if (pack_knapsack_meta) {
            packed_layers[l].min_weight = h_min_weight;
            packed_layers[l].single_parent_id = h_parent_id;
            packed_layers[l].single_parent_arc = h_parent_arc;
        }
    }

    thrust::host_vector<int> h_prev_sizes(root_nodes, 0);
    h_prev_sizes[root_idx] = 1;

    thrust::device_vector<int> d_prev_sizes = h_prev_sizes;
    thrust::device_vector<int> d_prev_offsets(root_nodes + 1, 0);
    thrust::exclusive_scan(d_prev_sizes.begin(), d_prev_sizes.end(), d_prev_offsets.begin());
    d_prev_offsets[root_nodes] = 1;

    thrust::device_vector<ObjType> d_prev_points(NOBJS, 0);
    bool warned_non_knapsack_dominance = false;

    for (int l = 1; l < bdd->num_layers; ++l) {
        const int prev_nodes = bdd->layers[l - 1].size();
        const int next_nodes = bdd->layers[l].size();
        if (d_prev_offsets.size() != static_cast<size_t>(prev_nodes + 1)) {
            set_reason(reason, "Previous offsets size does not match layer size");
            return NULL;
        }

        thrust::device_vector<int> d_next_sizes(next_nodes, 0);
        thrust::device_vector<int> d_next_offsets(next_nodes + 1, 0);
        thrust::device_vector<ObjType> d_next_points;

        PackedCudaLayer& packed = packed_layers[l];
        const int num_edges = packed.edge_src.size();
        if (num_edges > 0) {
            thrust::device_vector<int> d_edge_counts(num_edges, 0);
            thrust::device_vector<int> d_edge_offsets(num_edges + 1, 0);

            compute_edge_counts_kernel<<<ceil_div(num_edges, kThreadsPerBlock), kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(packed.edge_src.data()),
                thrust::raw_pointer_cast(d_prev_offsets.data()),
                thrust::raw_pointer_cast(d_edge_counts.data()),
                num_edges);
            if (!sync_kernel("compute_edge_counts_kernel", reason)) {
                return NULL;
            }

            thrust::exclusive_scan(d_edge_counts.begin(), d_edge_counts.end(), d_edge_offsets.begin());
            const int last_offset = d_edge_offsets[num_edges - 1];
            const int last_count = d_edge_counts[num_edges - 1];
            const int total_candidates = last_offset + last_count;
            d_edge_offsets[num_edges] = total_candidates;

            if (total_candidates > 0) {
                thrust::device_vector<ObjType> d_cand_points(total_candidates * NOBJS, 0);

                expand_candidates_points_kernel<<<ceil_div(num_edges, kThreadsPerBlock), kThreadsPerBlock>>>(
                    thrust::raw_pointer_cast(packed.edge_src.data()),
                    thrust::raw_pointer_cast(packed.edge_weights.data()),
                    thrust::raw_pointer_cast(d_edge_offsets.data()),
                    thrust::raw_pointer_cast(d_edge_counts.data()),
                    thrust::raw_pointer_cast(d_prev_offsets.data()),
                    thrust::raw_pointer_cast(d_prev_points.data()),
                    num_edges,
                    thrust::raw_pointer_cast(d_cand_points.data()));
                if (!sync_kernel("expand_candidates_points_kernel", reason)) {
                    return NULL;
                }

                thrust::device_vector<int> d_alive(total_candidates, 0);
                thrust::device_vector<int> d_cand_counts(next_nodes, 0);
                compute_dst_candidate_counts_kernel<<<ceil_div(next_nodes, kThreadsPerBlock), kThreadsPerBlock>>>(
                    thrust::raw_pointer_cast(packed.in_edge_offsets.data()),
                    thrust::raw_pointer_cast(d_edge_offsets.data()),
                    next_nodes,
                    thrust::raw_pointer_cast(d_cand_counts.data()));
                if (!sync_kernel("compute_dst_candidate_counts_kernel", reason)) {
                    return NULL;
                }

                const int max_seg_size = thrust::reduce(d_cand_counts.begin(),
                                                        d_cand_counts.end(),
                                                        0,
                                                        thrust::maximum<int>());
                if (max_seg_size > 0) {
                    if (problem_type == 2 && max_seg_size > kThreadsPerBlock) {
                        const int num_tiles = ceil_div(max_seg_size, kThreadsPerBlock);
                        dim3 grid(next_nodes, num_tiles);
                        mark_dominated_by_dst_tiled_kernel<<<grid, kThreadsPerBlock>>>(
                            thrust::raw_pointer_cast(d_cand_points.data()),
                            thrust::raw_pointer_cast(packed.in_edge_offsets.data()),
                            thrust::raw_pointer_cast(d_edge_offsets.data()),
                            next_nodes,
                            thrust::raw_pointer_cast(d_alive.data()),
                            thrust::raw_pointer_cast(d_next_sizes.data()));
                        if (!sync_kernel("mark_dominated_by_dst_tiled_kernel", reason)) {
                            return NULL;
                        }
                    } else {
                        mark_dominated_by_dst_single_block_kernel<<<next_nodes, kThreadsPerBlock>>>(
                            thrust::raw_pointer_cast(d_cand_points.data()),
                            thrust::raw_pointer_cast(packed.in_edge_offsets.data()),
                            thrust::raw_pointer_cast(d_edge_offsets.data()),
                            next_nodes,
                            thrust::raw_pointer_cast(d_alive.data()),
                            thrust::raw_pointer_cast(d_next_sizes.data()));
                        if (!sync_kernel("mark_dominated_by_dst_single_block_kernel", reason)) {
                            return NULL;
                        }
                    }
                }

                thrust::device_vector<int> d_alive_prefix(total_candidates, 0);
                thrust::exclusive_scan(d_alive.begin(), d_alive.end(), d_alive_prefix.begin());
                const int total_next = thrust::reduce(d_alive.begin(), d_alive.end(), 0);

                thrust::exclusive_scan(d_next_sizes.begin(), d_next_sizes.end(), d_next_offsets.begin());
                d_next_offsets[next_nodes] = total_next;

                d_next_points.resize(total_next * NOBJS);
                if (total_next > 0) {
                    scatter_alive_points_kernel<<<ceil_div(total_candidates, kThreadsPerBlock), kThreadsPerBlock>>>(
                        thrust::raw_pointer_cast(d_alive.data()),
                        thrust::raw_pointer_cast(d_alive_prefix.data()),
                        thrust::raw_pointer_cast(d_cand_points.data()),
                        thrust::raw_pointer_cast(d_next_points.data()),
                        total_candidates);
                    if (!sync_kernel("scatter_alive_points_kernel", reason)) {
                        return NULL;
                    }
                }
            }
        }

        if (dominance_strategy > 0) {
            LayerDominanceContext dom_ctx;
            dom_ctx.next_sizes = &d_next_sizes;
            dom_ctx.next_offsets = &d_next_offsets;
            dom_ctx.next_points = &d_next_points;
            dom_ctx.layer_min_weight = pack_knapsack_meta ? &packed.min_weight : NULL;
            dom_ctx.layer_single_parent_id = pack_knapsack_meta ? &packed.single_parent_id : NULL;
            dom_ctx.layer_single_parent_arc = pack_knapsack_meta ? &packed.single_parent_arc : NULL;
            dom_ctx.reason = reason;
            dom_ctx.warned_non_knapsack = &warned_non_knapsack_dominance;
            dom_ctx.failed = false;

            g_layer_dom_ctx = &dom_ctx;
            clock_t init = clock();
            BDDMultiObj::filter_dominance_cuda(bdd, l, problem_type, dominance_strategy, stats);
            if (stats != NULL) {
                stats->pareto_dominance_time += clock() - init;
            }
            g_layer_dom_ctx = NULL;

            if (dom_ctx.failed) {
                return NULL;
            }
        }

        d_prev_offsets.swap(d_next_offsets);
        d_prev_points.swap(d_next_points);
    }

    const int term_idx = bdd->get_terminal()->index;
    thrust::host_vector<int> h_offsets = d_prev_offsets;
    if (term_idx < 0 || term_idx + 1 >= h_offsets.size()) {
        set_reason(reason, "Invalid terminal index after CUDA enumeration");
        return NULL;
    }

    const int begin = h_offsets[term_idx];
    const int end = h_offsets[term_idx + 1];
    const int num_points = std::max(0, end - begin);

    ParetoFrontier* frontier = new ParetoFrontier;
    frontier->sols.resize(num_points * NOBJS, 0);
    if (num_points > 0) {
        thrust::host_vector<ObjType> h_points = d_prev_points;
        std::copy(h_points.begin() + begin * NOBJS,
                  h_points.begin() + end * NOBJS,
                  frontier->sols.begin());
    }

    if (reason != NULL) {
        reason->clear();
    }
    return frontier;
}
