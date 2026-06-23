// ----------------------------------------------------------
// BDD Multiobjective Algorithms - Implementation
// ----------------------------------------------------------

#include "bdd_multiobj.hpp"
#include "../enum/cpu/cpu_helpers.hpp"
#include "../enum/cpu/cpu_wrappers.hpp"
#include "../enum/gpu/cuda_wrappers.hpp"
#include <cstring>

// BDD CPU Drivers

ParetoFrontier* BDDMultiObj::pareto_frontier_topdown(BDD* bdd,
                                                     bool maximization,
                                                     const int problem_type,
                                                     const int state_dominance,
                                                     EnumerationStats* stats,
                                                     int cpu_threads,
                                                     int cpu_topdown_kernel) {
    return ::topdown_cpu_enumerate(bdd, maximization, problem_type, state_dominance, stats, cpu_threads, cpu_topdown_kernel);
}

ParetoFrontier* BDDMultiObj::pareto_frontier_bottomup(BDD* bdd,
                                                      bool maximization,
                                                      const int problem_type,
                                                      const int state_dominance,
                                                      EnumerationStats* stats,
                                                      int cpu_threads) {
    return ::bottomup_cpu_enumerate(bdd, maximization, problem_type, state_dominance, stats, cpu_threads);
}

ParetoFrontier* BDDMultiObj::pareto_frontier_dynamic_layer_cutset(BDD* bdd,
                                                                  bool maximization,
                                                                  const int problem_type,
                                                                  const int state_dominance,
                                                                  EnumerationStats* stats,
                                                                  int cpu_threads,
                                                                  int cpu_coupled_kernel) {
    return ::coupled_cpu_enumerate(bdd, maximization, problem_type, state_dominance, stats, cpu_threads, cpu_coupled_kernel);
}

// BDD CPU State Dominance & Completion

void BDDMultiObj::filter_dominance(BDD* bdd, const int layer, const int problem_type, const int state_dominance, EnumerationStats* stats) {
    ::filter_dominance_cpu(bdd, layer, problem_type, state_dominance, stats);
}

void BDDMultiObj::filter_dominance_knapsack(BDD* bdd, const int layer, EnumerationStats* stats) {
    ::filter_dominance_knapsack_cpu(bdd, layer, stats);
}

void BDDMultiObj::filter_dominance_setpacking(BDD* bdd, const int layer, EnumerationStats* stats) {
    ::filter_dominance_setpacking_cpu(bdd, layer, stats);
}

void BDDMultiObj::filter_completion(BDD* bdd, const int layer) {
    ::filter_completion_cpu(bdd, layer);
}

// MDD CPU Drivers

ParetoFrontier* BDDMultiObj::pareto_frontier_topdown(MDD* mdd, EnumerationStats* stats, int cpu_threads, int cpu_topdown_kernel) {
    return ::topdown_mdd_cpu_enumerate(mdd, stats, cpu_threads, cpu_topdown_kernel);
}

ParetoFrontier* BDDMultiObj::pareto_frontier_dynamic_layer_cutset(MDD* mdd, EnumerationStats* stats, int cpu_threads, int cpu_coupled_kernel) {
    return ::coupled_mdd_cpu_enumerate(mdd, stats, cpu_threads, cpu_coupled_kernel);
}

// GPU / CUDA Drivers & Wrappers

ParetoFrontier* BDDMultiObj::pareto_frontier_topdown_cuda(BDD* bdd, bool maximization, const int problem_type, const int state_dominance, EnumerationStats* stats, std::string* reason) {
    if (stats != NULL) {
        stats->cpu_state_dominance_s = 0.0;
        stats->dominance_filtered_total = 0;
        stats->layer_coupling = 0;
        reset_cpu_metrics_stats(stats);
    }
    return ::topdown_cuda_enumerate(bdd, maximization, problem_type, state_dominance, stats, reason);
}

ParetoFrontier* BDDMultiObj::pareto_frontier_topdown_cuda(MDD* mdd, EnumerationStats* stats, std::string* reason) {
    if (stats != NULL) {
        stats->cpu_state_dominance_s = 0.0;
        stats->dominance_filtered_total = 0;
        stats->layer_coupling = 0;
        reset_cpu_metrics_stats(stats);
    }
    return ::topdown_mdd_cuda_enumerate(mdd, stats, reason);
}

ParetoFrontier* BDDMultiObj::pareto_frontier_dynamic_layer_cutset_cuda(MDD* mdd, EnumerationStats* stats, std::string* reason) {
    if (stats != NULL) {
        stats->cpu_state_dominance_s = 0.0;
        stats->dominance_filtered_total = 0;
        stats->layer_coupling = 0;
        reset_cpu_metrics_stats(stats);
    }
    return ::coupled_cuda_enumerate(mdd, stats, reason);
}
