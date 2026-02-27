// ----------------------------------------------------------
// BDD Multiobjective Algorithms
// ----------------------------------------------------------

#ifndef BDD_MULTIOBJ_HPP_
#define BDD_MULTIOBJ_HPP_

#include <string>

#include "../mdd/mdd.hpp"
#include "../util/util.hpp"
#include "bdd.hpp"
#include "pareto_frontier.hpp"


//
// Multiobjective stats
//
struct MultiObjectiveStats {
    // Time spent in pareto dominance filtering
    clock_t pareto_dominance_time;
    // Solutions filtered by pareto dominance
    int pareto_dominance_filtered;
    // Layer where coupling happened
    int layer_coupling;
    // Enable lightweight CPU performance aggregation
    bool cpu_perf_enabled;
    // Aggregated wall times (seconds) for CPU phases
    double cpu_expand_td_wall_s;
    double cpu_expand_bu_wall_s;
    double cpu_recompute_td_wall_s;
    double cpu_recompute_bu_wall_s;
    double cpu_dominance_wall_s;
    double cpu_cutset_sort_wall_s;
    double cpu_cutset_convolution_wall_s;
    double cpu_cutset_partial_merge_wall_s;
    // Aggregated counters for CPU phases
    int cpu_layers_td;
    int cpu_layers_bu;
    long long cpu_nodes_expanded;
    int cpu_cutset_size;

    // Constructor
    MultiObjectiveStats() 
    : pareto_dominance_time(0),
      pareto_dominance_filtered(0),
      layer_coupling(0),
      cpu_perf_enabled(false),
      cpu_expand_td_wall_s(0.0),
      cpu_expand_bu_wall_s(0.0),
      cpu_recompute_td_wall_s(0.0),
      cpu_recompute_bu_wall_s(0.0),
      cpu_dominance_wall_s(0.0),
      cpu_cutset_sort_wall_s(0.0),
      cpu_cutset_convolution_wall_s(0.0),
      cpu_cutset_partial_merge_wall_s(0.0),
      cpu_layers_td(0),
      cpu_layers_bu(0),
      cpu_nodes_expanded(0),
      cpu_cutset_size(0)
    { }
};


//
// BDD Multiobjective Algorithms
//
struct BDDMultiObj {
    // Find pareto frontier from top-down approach
	static ParetoFrontier* pareto_frontier_topdown(BDD* bdd, bool maximization=true, const int problem_type=-1, const int dominance_strategy=0, MultiObjectiveStats* stats = NULL, int cpu_threads = 1);

    // Find pareto frontier from top-down approach / CUDA
    // kernel_version:
    //   1 = one block per node
    //   2 = fixed number of blocks per node (2D grid)
    //   3 = dynamic number of blocks per node (1D grid + binary-search destination lookup)
    static ParetoFrontier* pareto_frontier_topdown_cuda(BDD* bdd, bool maximization=true, const int problem_type=-1, const int dominance_strategy=0, MultiObjectiveStats* stats = NULL, std::string* reason = NULL, int kernel_version = 3);

    // Find pareto frontier from top-down approach / CUDA for MDD
    static ParetoFrontier* pareto_frontier_topdown_cuda(MDD* mdd, MultiObjectiveStats* stats = NULL, std::string* reason = NULL, int kernel_version = 3);

    // Filter layer based on dominance / CUDA
    static void filter_dominance_cuda(BDD* bdd, const int layer, const int problem_type, const int dominance_strategy, MultiObjectiveStats* stats);

    // Filter layer based on dominance / knapsack / CUDA
    static void filter_dominance_knapsack_cuda(BDD* bdd, const int layer, MultiObjectiveStats* stats);

    // Find pareto frontier from bottom-up approach
	static ParetoFrontier* pareto_frontier_bottomup(BDD* bdd, bool maximization=true, const int problem_type=-1, const int dominance_strategy=0, MultiObjectiveStats* stats = NULL, int cpu_threads = 1);

    // Find pareto frontier using dynamic layer cutset
    static ParetoFrontier* pareto_frontier_dynamic_layer_cutset(BDD* bdd, bool maximization=true, const int problem_type=-1, const int dominance_strategy=0, MultiObjectiveStats* stats = NULL, int cpu_threads = 1);

    // Approximate pareto frontier / top-down
    static void approximate_pareto_frontier_topdown(BDD* bdd, const int s_max, const int t_max);

    // Approximate pareto frontier with dominance filtering / top-down
    static void approximate_pareto_frontier_topdown_dominance(BDD* bdd, const int s_max, const int t_max);
    
    // Approximate pareto frontier / bottom-up
    static void approximate_pareto_frontier_bottomup(BDD* bdd, const int s_max, const int t_max);

    // Filter layer based on dominance
    static void filter_dominance(BDD* bdd, const int layer, const int problem_type, const int dominance_strategy, MultiObjectiveStats* stats);
    
    // Filter layer based on dominance / knapsack
    static void filter_dominance_knapsack(BDD* bdd, const int layer, MultiObjectiveStats* stats);
    
    // Filter layer based on dominance / set packing
    static void filter_dominance_setpacking(BDD* bdd, const int layer, MultiObjectiveStats* stats);
    
    // Filter layer based on dominance / set covering
    static void filter_dominance_setcovering(BDD* bdd, const int layer, MultiObjectiveStats* stats);

    // Filter layer based on dominance during approximation / knapsack
    static void filter_dominance_knapsack_approx(BDD* bdd, const int layer);
    
    // Filter layer based on node completion
    static void filter_completion(BDD* bdd, const int layer);    

    // Find pareto frontier from top-down approach - MDD version
	static ParetoFrontier* pareto_frontier_topdown(MDD* bdd, MultiObjectiveStats* stats, int cpu_threads = 1);

    // Find pareto frontier using dynamic layer cutset
    static ParetoFrontier* pareto_frontier_dynamic_layer_cutset(MDD* mdd, MultiObjectiveStats* stats, int cpu_threads = 1);

    // Find pareto frontier using dynamic layer cutset / CUDA
    static ParetoFrontier* pareto_frontier_dynamic_layer_cutset_cuda(MDD* mdd, MultiObjectiveStats* stats = NULL, std::string* reason = NULL, int kernel_version = 3);
};



#endif 
