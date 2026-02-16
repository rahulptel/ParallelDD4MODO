// ----------------------------------------------------------
// CUDA Top-Down Enumeration for BDD - Implementation
// ----------------------------------------------------------

#include "topdown_cuda.hpp"

#include <algorithm>
#include <vector>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace {

constexpr int kThreadsPerBlock = 128;

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

__global__ void expand_candidates_kernel(const int* edge_src,
                                         const int* edge_dst,
                                         const ObjType* edge_weights,
                                         const int* edge_offsets,
                                         const int* edge_counts,
                                         const int* prev_offsets,
                                         const ObjType* prev_points,
                                         int num_edges,
                                         int* cand_dst,
                                         ObjType* cand_points) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) {
        return;
    }

    const int src = edge_src[e];
    const int dst = edge_dst[e];
    const int src_begin = prev_offsets[src];
    const int out_begin = edge_offsets[e];
    const int count = edge_counts[e];
    const ObjType* w = edge_weights + (e * NOBJS);

    for (int k = 0; k < count; ++k) {
        const int src_idx = src_begin + k;
        const int out_idx = out_begin + k;

        cand_dst[out_idx] = dst;
        for (int o = 0; o < NOBJS; ++o) {
            cand_points[out_idx * NOBJS + o] = prev_points[src_idx * NOBJS + o] + w[o];
        }
    }
}

__global__ void gather_points_kernel(const int* order,
                                     const ObjType* in_points,
                                     ObjType* out_points,
                                     int num_points) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) {
        return;
    }

    const int src = order[i];
    for (int o = 0; o < NOBJS; ++o) {
        out_points[i * NOBJS + o] = in_points[src * NOBJS + o];
    }
}

__global__ void mark_dominated_segments_kernel(const ObjType* points,
                                               const int* seg_offsets,
                                               const int* seg_counts,
                                               int num_segments,
                                               int* alive) {
    const int seg = blockIdx.x;
    if (seg >= num_segments) {
        return;
    }

    const int begin = seg_offsets[seg];
    const int len = seg_counts[seg];

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

            for (int o = 0; o < NOBJS; ++o) {
                const ObjType a = points[j * NOBJS + o];
                const ObjType b = points[i * NOBJS + o];
                ge_all = ge_all && (a >= b);
                strict = strict || (a > b);
            }

            // Deterministic tie-break for duplicate points: keep lower local index.
            if (ge_all && (strict || (local_j < local_i))) {
                dominated = true;
            }
        }

        alive[i] = dominated ? 0 : 1;
    }
}

__global__ void write_segment_sizes_kernel(const int* unique_dst,
                                           const int* seg_offsets,
                                           const int* seg_counts,
                                           const int* alive,
                                           int num_segments,
                                           int* next_sizes) {
    const int seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= num_segments) {
        return;
    }

    const int begin = seg_offsets[seg];
    const int len = seg_counts[seg];
    int live = 0;
    for (int i = 0; i < len; ++i) {
        live += alive[begin + i];
    }
    next_sizes[unique_dst[seg]] = live;
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

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

} // namespace


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


ParetoFrontier* topdown_cuda_enumerate(BDD* bdd, bool maximization, std::string* reason) {
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

    thrust::host_vector<int> h_prev_sizes(root_nodes, 0);
    h_prev_sizes[root_idx] = 1;

    thrust::device_vector<int> d_prev_sizes = h_prev_sizes;
    thrust::device_vector<int> d_prev_offsets(root_nodes + 1, 0);
    thrust::exclusive_scan(d_prev_sizes.begin(), d_prev_sizes.end(), d_prev_offsets.begin());
    d_prev_offsets[root_nodes] = 1;

    thrust::device_vector<ObjType> d_prev_points(NOBJS, 0);

    for (int l = 1; l < bdd->num_layers; ++l) {
        const int prev_nodes = bdd->layers[l - 1].size();
        const int next_nodes = bdd->layers[l].size();
        if (d_prev_offsets.size() != static_cast<size_t>(prev_nodes + 1)) {
            set_reason(reason, "Previous offsets size does not match layer size");
            return NULL;
        }

        std::vector<int> h_edge_src;
        std::vector<int> h_edge_dst;
        std::vector<ObjType> h_edge_weights;

        const int first_arc_type = maximization ? 1 : 0;
        const int second_arc_type = maximization ? 0 : 1;

        for (int dst_idx = 0; dst_idx < next_nodes; ++dst_idx) {
            Node* dst_node = bdd->layers[l][dst_idx];
            const int arc_order[2] = {first_arc_type, second_arc_type};
            for (int arc_pos = 0; arc_pos < 2; ++arc_pos) {
                const int arc_type = arc_order[arc_pos];
                for (std::vector<Node*>::iterator it = dst_node->prev[arc_type].begin();
                     it != dst_node->prev[arc_type].end(); ++it) {
                    Node* src_node = *it;
                    h_edge_src.push_back(src_node->index);
                    h_edge_dst.push_back(dst_idx);

                    ObjType* w = src_node->weights[arc_type];
                    for (int o = 0; o < NOBJS; ++o) {
                        h_edge_weights.push_back(w != NULL ? w[o] : 0);
                    }
                }
            }
        }

        thrust::device_vector<int> d_next_sizes(next_nodes, 0);
        thrust::device_vector<int> d_next_offsets(next_nodes + 1, 0);
        thrust::device_vector<ObjType> d_next_points;

        const int num_edges = h_edge_src.size();
        if (num_edges > 0) {
            thrust::device_vector<int> d_edge_src = h_edge_src;
            thrust::device_vector<int> d_edge_dst = h_edge_dst;
            thrust::device_vector<ObjType> d_edge_weights = h_edge_weights;

            thrust::device_vector<int> d_edge_counts(num_edges, 0);
            thrust::device_vector<int> d_edge_offsets(num_edges, 0);

            compute_edge_counts_kernel<<<ceil_div(num_edges, kThreadsPerBlock), kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(d_edge_src.data()),
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

            if (total_candidates > 0) {
                thrust::device_vector<int> d_cand_dst(total_candidates, 0);
                thrust::device_vector<ObjType> d_cand_points(total_candidates * NOBJS, 0);

                expand_candidates_kernel<<<ceil_div(num_edges, kThreadsPerBlock), kThreadsPerBlock>>>(
                    thrust::raw_pointer_cast(d_edge_src.data()),
                    thrust::raw_pointer_cast(d_edge_dst.data()),
                    thrust::raw_pointer_cast(d_edge_weights.data()),
                    thrust::raw_pointer_cast(d_edge_offsets.data()),
                    thrust::raw_pointer_cast(d_edge_counts.data()),
                    thrust::raw_pointer_cast(d_prev_offsets.data()),
                    thrust::raw_pointer_cast(d_prev_points.data()),
                    num_edges,
                    thrust::raw_pointer_cast(d_cand_dst.data()),
                    thrust::raw_pointer_cast(d_cand_points.data()));
                if (!sync_kernel("expand_candidates_kernel", reason)) {
                    return NULL;
                }

                thrust::device_vector<int> d_order(total_candidates, 0);
                thrust::sequence(d_order.begin(), d_order.end());
                thrust::sort_by_key(d_cand_dst.begin(), d_cand_dst.end(), d_order.begin());

                thrust::device_vector<ObjType> d_sorted_points(total_candidates * NOBJS, 0);
                gather_points_kernel<<<ceil_div(total_candidates, kThreadsPerBlock), kThreadsPerBlock>>>(
                    thrust::raw_pointer_cast(d_order.data()),
                    thrust::raw_pointer_cast(d_cand_points.data()),
                    thrust::raw_pointer_cast(d_sorted_points.data()),
                    total_candidates);
                if (!sync_kernel("gather_points_kernel", reason)) {
                    return NULL;
                }

                thrust::device_vector<int> d_unique_dst(total_candidates, 0);
                thrust::device_vector<int> d_seg_counts(total_candidates, 0);
                typedef thrust::device_vector<int>::iterator It;
                thrust::pair<It, It> seg_end = thrust::reduce_by_key(
                    d_cand_dst.begin(),
                    d_cand_dst.end(),
                    thrust::make_constant_iterator(1),
                    d_unique_dst.begin(),
                    d_seg_counts.begin());

                const int num_segments = seg_end.first - d_unique_dst.begin();
                d_unique_dst.resize(num_segments);
                d_seg_counts.resize(num_segments);

                if (num_segments > 0) {
                    thrust::device_vector<int> d_seg_offsets(num_segments + 1, 0);
                    thrust::exclusive_scan(d_seg_counts.begin(), d_seg_counts.end(), d_seg_offsets.begin());
                    d_seg_offsets[num_segments] = total_candidates;

                    thrust::device_vector<int> d_alive(total_candidates, 0);
                    mark_dominated_segments_kernel<<<num_segments, kThreadsPerBlock>>>(
                        thrust::raw_pointer_cast(d_sorted_points.data()),
                        thrust::raw_pointer_cast(d_seg_offsets.data()),
                        thrust::raw_pointer_cast(d_seg_counts.data()),
                        num_segments,
                        thrust::raw_pointer_cast(d_alive.data()));
                    if (!sync_kernel("mark_dominated_segments_kernel", reason)) {
                        return NULL;
                    }

                    thrust::device_vector<int> d_alive_prefix(total_candidates, 0);
                    thrust::exclusive_scan(d_alive.begin(), d_alive.end(), d_alive_prefix.begin());
                    const int total_next = thrust::reduce(d_alive.begin(), d_alive.end(), 0);

                    write_segment_sizes_kernel<<<ceil_div(num_segments, kThreadsPerBlock), kThreadsPerBlock>>>(
                        thrust::raw_pointer_cast(d_unique_dst.data()),
                        thrust::raw_pointer_cast(d_seg_offsets.data()),
                        thrust::raw_pointer_cast(d_seg_counts.data()),
                        thrust::raw_pointer_cast(d_alive.data()),
                        num_segments,
                        thrust::raw_pointer_cast(d_next_sizes.data()));
                    if (!sync_kernel("write_segment_sizes_kernel", reason)) {
                        return NULL;
                    }

                    thrust::exclusive_scan(d_next_sizes.begin(), d_next_sizes.end(), d_next_offsets.begin());
                    d_next_offsets[next_nodes] = total_next;

                    d_next_points.resize(total_next * NOBJS);
                    if (total_next > 0) {
                        scatter_alive_points_kernel<<<ceil_div(total_candidates, kThreadsPerBlock), kThreadsPerBlock>>>(
                            thrust::raw_pointer_cast(d_alive.data()),
                            thrust::raw_pointer_cast(d_alive_prefix.data()),
                            thrust::raw_pointer_cast(d_sorted_points.data()),
                            thrust::raw_pointer_cast(d_next_points.data()),
                            total_candidates);
                        if (!sync_kernel("scatter_alive_points_kernel", reason)) {
                            return NULL;
                        }
                    }
                }
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
