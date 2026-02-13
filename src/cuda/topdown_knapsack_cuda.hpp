// ----------------------------------------------------------
// CUDA Top-Down Enumeration for Knapsack BDD
// ----------------------------------------------------------

#ifndef TOPDOWN_KNAPSACK_CUDA_HPP_
#define TOPDOWN_KNAPSACK_CUDA_HPP_

#include <string>

#include "../bdd/bdd.hpp"
#include "../bdd/pareto_frontier.hpp"

// Checks whether at least one CUDA device is available.
bool topdown_knapsack_cuda_available(std::string* reason);

// Runs top-down frontier enumeration on CUDA for knapsack BDDs.
// Returns NULL on failure and fills reason when provided.
ParetoFrontier* topdown_knapsack_cuda_enumerate(BDD* bdd, std::string* reason);

#endif
