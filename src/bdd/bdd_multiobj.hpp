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
#include "../util/stats.hpp"

//
// BDD Multiobjective Algorithms
//
struct BDDMultiObj {
    // Find pareto frontier from top-down approach
	static ParetoFrontier* pareto_frontier_topdown(BDD* bdd, bool maximization=true, const int problem_type=-1, const int dominance_strategy=0, EnumerationStats* stats = NULL, int cpu_threads = 1);

    // Find pareto frontier from top-down approach / CUDA
    // kernel_version:
    //   1 = one block per node
    //   2 = fixed number of blocks per node (2D grid)
    //   3 = dynamic number of blocks per node (1D grid + binary-search destination lookup)
    static ParetoFrontier* pareto_frontier_topdown_cuda(BDD* bdd, bool maximization=true, const int problem_type=-1, const int dominance_strategy=0, EnumerationStats* stats = NULL, std::string* reason = NULL, int kernel_version = 3);

    // Find pareto frontier from top-down approach / CUDA for MDD
    static ParetoFrontier* pareto_frontier_topdown_cuda(MDD* mdd, EnumerationStats* stats = NULL, std::string* reason = NULL, int kernel_version = 3);

    // Filter layer based on dominance / CUDA
    static void filter_dominance_cuda(BDD* bdd, const int layer, const int problem_type, const int dominance_strategy, EnumerationStats* stats);

    // Filter layer based on dominance / knapsack / CUDA
    static void filter_dominance_knapsack_cuda(BDD* bdd, const int layer, EnumerationStats* stats);

    // Find pareto frontier from bottom-up approach
	static ParetoFrontier* pareto_frontier_bottomup(BDD* bdd, bool maximization=true, const int problem_type=-1, const int dominance_strategy=0, EnumerationStats* stats = NULL, int cpu_threads = 1);

    // Find pareto frontier using dynamic layer cutset
    static ParetoFrontier* pareto_frontier_dynamic_layer_cutset(BDD* bdd, bool maximization=true, const int problem_type=-1, const int dominance_strategy=0, EnumerationStats* stats = NULL, int cpu_threads = 1);

    // Filter layer based on dominance
    static void filter_dominance(BDD* bdd, const int layer, const int problem_type, const int dominance_strategy, EnumerationStats* stats);
    
    // Filter layer based on dominance / knapsack
    static void filter_dominance_knapsack(BDD* bdd, const int layer, EnumerationStats* stats);
    
    // Filter layer based on dominance / set packing
    static void filter_dominance_setpacking(BDD* bdd, const int layer, EnumerationStats* stats);
    
    // Filter layer based on node completion
    static void filter_completion(BDD* bdd, const int layer);    

    // Find pareto frontier from top-down approach - MDD version
	static ParetoFrontier* pareto_frontier_topdown(MDD* bdd, EnumerationStats* stats, int cpu_threads = 1);

    // Find pareto frontier using dynamic layer cutset
    static ParetoFrontier* pareto_frontier_dynamic_layer_cutset(MDD* mdd, EnumerationStats* stats, int cpu_threads = 1);

    // Find pareto frontier using dynamic layer cutset / CUDA
    static ParetoFrontier* pareto_frontier_dynamic_layer_cutset_cuda(MDD* mdd, EnumerationStats* stats = NULL, std::string* reason = NULL, int kernel_version = 3);
};



#endif 
