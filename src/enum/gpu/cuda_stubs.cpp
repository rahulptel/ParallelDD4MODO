// Stub implementations of CUDA wrappers used when compiling without USE_CUDA

#include "cuda_wrappers.hpp"
#include "../pareto_frontier.hpp"
#include "../../mdd/mdd.hpp"
#include "../../bdd/bdd.hpp"
#include "../multiobj_enum.hpp"

ParetoFrontier* topdown_cuda_enumerate(BDD* bdd, bool maximization, const int problem_type, const int state_dominance, EnumerationStats* stats, std::string* reason) {
    (void)bdd;
    (void)maximization;
    (void)problem_type;
    (void)state_dominance;
    (void)stats;
    if (reason != NULL) {
        *reason = "GPU backend requested but binary was built without CUDA support";
    }
    return NULL;
}

ParetoFrontier* topdown_mdd_cuda_enumerate(MDD* mdd, EnumerationStats* stats, std::string* reason) {
    (void)mdd;
    (void)stats;
    if (reason != NULL) {
        *reason = "GPU backend requested but binary was built without CUDA support";
    }
    return NULL;
}

ParetoFrontier* coupled_cuda_enumerate(MDD* mdd, EnumerationStats* stats, std::string* reason) {
    (void)mdd;
    (void)stats;
    if (reason != NULL) {
        *reason = "GPU backend requested but binary was built without CUDA support";
    }
    return NULL;
}

ParetoFrontier* coupled_bdd_cuda_enumerate(BDD* bdd, bool maximization, const int problem_type, const int state_dominance, EnumerationStats* stats, std::string* reason) {
    (void)bdd;
    (void)maximization;
    (void)problem_type;
    (void)state_dominance;
    (void)stats;
    if (reason != NULL) {
        *reason = "GPU backend requested but binary was built without CUDA support";
    }
    return NULL;
}
