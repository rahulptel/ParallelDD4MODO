// ----------------------------------------------------------
// Shared CUDA enumeration types and layer expansion API
// ----------------------------------------------------------

#ifndef CUDA_ENUM_TYPES_CUH_
#define CUDA_ENUM_TYPES_CUH_

#include <string>

#include <thrust/device_vector.h>

#include "../util/util.hpp"

struct EnumerationStats;

// Flat representation of one MDD layer's connectivity, on device.
struct PackedMDDLayer {
    int num_nodes;

    // Top-down: incoming arcs grouped by destination node (this layer).
    // edge_src indices refer to nodes in layer-1.
    int td_num_edges;
    thrust::device_vector<int> td_in_edge_offsets;
    thrust::device_vector<int> td_edge_src;
    thrust::device_vector<ObjType> td_edge_weights;

    // Bottom-up: outgoing arcs grouped by node (this layer).
    // edge_src indices refer to head nodes in layer+1.
    int bu_num_edges;
    thrust::device_vector<int> bu_in_edge_offsets;
    thrust::device_vector<int> bu_edge_src;
    thrust::device_vector<ObjType> bu_edge_weights;

    // Per-node arc counts for the layer-value heuristic.
    thrust::device_vector<int> out_arc_counts;
    thrust::device_vector<int> in_arc_counts;
};

int compute_expansion_score(const thrust::device_vector<int>& offsets,
                        const thrust::device_vector<int>& arc_counts,
                        int num_nodes);

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
    long long* total_candidates_out = NULL,
    long long* total_next_out = NULL,
    double* std_candidates_out = NULL,
    double* std_survivors_out = NULL,
    long long* gpu_mem_baseline_used_bytes = NULL,
    long long* gpu_mem_peak_used_bytes = NULL,
    long long* gpu_mem_peak_reserved_bytes = NULL);

#endif
