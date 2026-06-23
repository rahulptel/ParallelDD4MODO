#pragma once

#include <string>

#include "../multiobj_enum.hpp"
#include "../../mdd/mdd.hpp"
#include "../../bdd/bdd.hpp"
#include "../pareto_frontier.hpp"

// Declarations of CUDA functions that are either real or stubs depending on ENABLE_CUDA.
ParetoFrontier* topdown_cuda_enumerate(BDD* bdd, bool maximization, const int problem_type, const int state_dominance, EnumerationStats* stats, std::string* reason);
ParetoFrontier* topdown_mdd_cuda_enumerate(MDD* mdd, EnumerationStats* stats, std::string* reason);
ParetoFrontier* coupled_cuda_enumerate(MDD* mdd, EnumerationStats* stats, std::string* reason);
ParetoFrontier* coupled_bdd_cuda_enumerate(BDD* bdd, bool maximization, const int problem_type, const int state_dominance, EnumerationStats* stats, std::string* reason);
