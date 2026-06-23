// ----------------------------------------------------------
// CPU Bottom-Up BFS Kernels - Implementation
// ----------------------------------------------------------

#include "cpu_helpers.hpp"
#include "cpu_wrappers.hpp"

#include <vector>
#include <algorithm>
#include <cassert>
#include <limits>

void expand_layer_bottomup(BDD* bdd, const int l, const bool maximization, ParetoFrontierManager* mgmr, const int cpu_threads) {
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);
    if (maximization) {
        const int layer_size = bdd->layers[l].size();
        CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
        for (int i = 0; i < layer_size; ++i) {
            Node* node = bdd->layers[l][i];

            // Request frontier
            node->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);

            // add outgoing one arcs
            if (node->arcs[1] != NULL) {
                node->pareto_frontier_bu->merge(*(node->arcs[1]->pareto_frontier_bu), node->weights[1]);
            }

            // add outgoing zero arcs
            if (node->arcs[0] != NULL) {
                node->pareto_frontier_bu->merge(*(node->arcs[0]->pareto_frontier_bu), node->weights[0]);
            }
        }
    } else {
        const int layer_size = bdd->layers[l].size();
        CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
        for (int i = 0; i < layer_size; ++i) {
            Node* node = bdd->layers[l][i];

            // Request frontier
            node->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);

            // add outgoing zero arcs
            if (node->arcs[0] != NULL) {
                node->pareto_frontier_bu->merge(*(node->arcs[0]->pareto_frontier_bu), node->weights[0]);
            }

            // add outgoing one arcs
            if (node->arcs[1] != NULL) {
                node->pareto_frontier_bu->merge(*(node->arcs[1]->pareto_frontier_bu), node->weights[1]);
            }
        }
    }
    // deallocate next layer
    for (size_t i = 0; i < bdd->layers[l+1].size(); ++i) {
        recycle_frontier(mgmr, bdd->layers[l+1][i]->pareto_frontier_bu, parallel_mode);
    }
}

void expand_layer_bottomup_cpu_kernel1_mdd(MDD* mdd,
                                           const int l,
                                           ParetoFrontierManager* mgmr,
                                           const bool parallel_mode,
                                           const int threads) {
    const int layer_size = mdd->layers[l].size();
    CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
    for (int i = 0; i < layer_size; ++i) {
        MDDNode* node = mdd->layers[l][i];
        node->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);

        for (MDDArc* arc : node->out_arcs_list) {
            node->pareto_frontier_bu->merge(*(arc->head->pareto_frontier_bu), arc->weights);
        }
    }

    for (size_t i = 0; i < mdd->layers[l+1].size(); ++i) {
        recycle_frontier(mgmr, mdd->layers[l+1][i]->pareto_frontier_bu, parallel_mode);
        delete mdd->layers[l+1][i];
    }
}

bool expand_layer_bottomup_cpu_kernel3_mdd(MDD* mdd,
                                           const int l,
                                           ParetoFrontierManager* mgmr,
                                           const bool parallel_mode,
                                           const int threads) {
    const int layer_size = mdd->layers[l].size();
    const size_t kMaxSizeT = std::numeric_limits<size_t>::max();
    std::vector<size_t> node_offsets(layer_size + 1, 0);

    for (int i = 0; i < layer_size; ++i) {
        size_t node_candidates = 0;
        MDDNode* node = mdd->layers[l][i];
        for (MDDArc* arc : node->out_arcs_list) {
            if (arc == NULL || arc->head == NULL || arc->head->pareto_frontier_bu == NULL) {
                continue;
            }
            const size_t arc_candidates = static_cast<size_t>(arc->head->pareto_frontier_bu->get_num_sols());
            if (node_candidates > kMaxSizeT - arc_candidates) {
                return false;
            }
            node_candidates += arc_candidates;
        }
        if (node_offsets[i] > kMaxSizeT - node_candidates) {
            return false;
        }
        node_offsets[i + 1] = node_offsets[i] + node_candidates;
    }

    const size_t total_candidates = node_offsets[layer_size];
    if (NOBJS > 0 && total_candidates > kMaxSizeT / static_cast<size_t>(NOBJS)) {
        return false;
    }
    std::vector<ObjType> cand_points(total_candidates * NOBJS, 0);

    CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
    for (int i = 0; i < layer_size; ++i) {
        MDDNode* node = mdd->layers[l][i];
        size_t write_idx = node_offsets[i];
        for (MDDArc* arc : node->out_arcs_list) {
            if (arc == NULL || arc->head == NULL || arc->head->pareto_frontier_bu == NULL) {
                continue;
            }
            const std::vector<ObjType>& next_sols = arc->head->pareto_frontier_bu->sols;
            const size_t num_next_sols = next_sols.size() / NOBJS;
            const ObjType* shift = arc->weights;
            for (size_t s = 0; s < num_next_sols; ++s) {
                const size_t out_pos = (write_idx + s) * NOBJS;
                const size_t in_pos = s * NOBJS;
                for (int o = 0; o < NOBJS; ++o) {
                    cand_points[out_pos + o] = next_sols[in_pos + o] + shift[o];
                }
            }
            write_idx += num_next_sols;
        }
        assert(write_idx == node_offsets[i + 1]);
    }

    const size_t kCpuTilePoints = 256;
    std::vector<CpuTopdownTileTask> tasks;
    tasks.reserve((total_candidates + kCpuTilePoints - 1) / kCpuTilePoints + static_cast<size_t>(layer_size));
    for (int i = 0; i < layer_size; ++i) {
        const size_t node_begin = node_offsets[i];
        const size_t node_end = node_offsets[i + 1];
        const size_t node_len = node_end - node_begin;
        for (size_t local_begin = 0; local_begin < node_len; local_begin += kCpuTilePoints) {
            CpuTopdownTileTask task;
            task.node_idx = i;
            task.local_begin = local_begin;
            task.local_end = std::min(node_len, local_begin + kCpuTilePoints);
            tasks.push_back(task);
        }
    }

    std::vector<unsigned char> alive(total_candidates, 1);
    const long long task_count = static_cast<long long>(tasks.size());
    CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
    for (long long t = 0; t < task_count; ++t) {
        const CpuTopdownTileTask& task = tasks[static_cast<size_t>(t)];
        const size_t node_begin = node_offsets[task.node_idx];
        const size_t node_end = node_offsets[task.node_idx + 1];
        const size_t tile_begin = node_begin + task.local_begin;
        const size_t tile_end = node_begin + task.local_end;

        for (size_t idx_i = tile_begin; idx_i < tile_end; ++idx_i) {
            bool dominated = false;
            const size_t pos_i = idx_i * NOBJS;
            for (size_t idx_j = node_begin; idx_j < node_end && !dominated; ++idx_j) {
                if (idx_i == idx_j) {
                    continue;
                }
                const size_t pos_j = idx_j * NOBJS;
                bool ge_all = true;
                bool strict = false;
                for (int o = 0; o < NOBJS; ++o) {
                    const ObjType a = cand_points[pos_j + o];
                    const ObjType b = cand_points[pos_i + o];
                    ge_all = ge_all && (a >= b);
                    strict = strict || (a > b);
                }
                if (ge_all && (strict || (idx_j < idx_i))) {
                    dominated = true;
                }
            }
            alive[idx_i] = dominated ? 0 : 1;
        }
    }

    CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
    for (int i = 0; i < layer_size; ++i) {
        MDDNode* node = mdd->layers[l][i];
        ParetoFrontier* frontier = request_frontier(mgmr, parallel_mode);
        const size_t node_begin = node_offsets[i];
        const size_t node_end = node_offsets[i + 1];
        for (size_t idx = node_begin; idx < node_end; ++idx) {
            if (alive[idx] == 0) {
                continue;
            }
            const size_t pos = idx * NOBJS;
            for (int o = 0; o < NOBJS; ++o) {
                frontier->sols.push_back(cand_points[pos + o]);
            }
        }
        node->pareto_frontier_bu = frontier;
    }

    for (size_t i = 0; i < mdd->layers[l+1].size(); ++i) {
        recycle_frontier(mgmr, mdd->layers[l+1][i]->pareto_frontier_bu, parallel_mode);
        delete mdd->layers[l+1][i];
    }

    return true;
}
