// ----------------------------------------------------------
// CUDA Top-Down Enumeration for BDD
// ----------------------------------------------------------

#ifndef TOPDOWN_CUDA_HPP_
#define TOPDOWN_CUDA_HPP_

#include <string>

#include "../bdd/bdd.hpp"
#include "../bdd/pareto_frontier.hpp"

// Checks whether at least one CUDA device is available.
bool topdown_cuda_available(std::string* reason);

// Runs top-down frontier enumeration on CUDA for BDDs.
// Returns NULL on failure and fills reason when provided.
ParetoFrontier* topdown_cuda_enumerate(BDD* bdd, bool maximization, std::string* reason);

#endif
