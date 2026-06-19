// ----------------------------------------------------------
// CUDA enumeration public entry points
// ----------------------------------------------------------

#include "cuda_wrappers.hpp"
#include "enum_types.cuh"

#include <cuda_runtime.h>

ParetoFrontier* enumerate_bdd_topdown(BDD* bdd,
                                                bool maximization,
                                                const int problem_type,
                                                const int state_dominance,
                                                EnumerationStats* stats,
                                                std::string* reason);

ParetoFrontier* enumerate_mdd_topdown(MDD* mdd,
                                                EnumerationStats* stats,
                                                std::string* reason);

ParetoFrontier* enumerate_mdd_coupled(MDD* mdd,
                                            EnumerationStats* stats,
                                            std::string* reason);

namespace {

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

bool cuda_enumeration_available(std::string* reason) {
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

bool prepare_cuda_device(std::string* reason) {
    if (!cuda_enumeration_available(reason)) {
        return false;
    }
    return cuda_ok(cudaSetDevice(0), "cudaSetDevice", reason);
}

} // namespace

bool topdown_cuda_available(std::string* reason) {
    return cuda_enumeration_available(reason);
}

bool coupled_cuda_available(std::string* reason) {
    return cuda_enumeration_available(reason);
}

ParetoFrontier* topdown_cuda_enumerate(BDD* bdd,
                                       bool maximization,
                                       const int problem_type,
                                       const int state_dominance,
                                       EnumerationStats* stats,
                                       std::string* reason) {
    if (!prepare_cuda_device(reason)) {
        return NULL;
    }
    return enumerate_bdd_topdown(bdd, maximization, problem_type, state_dominance, stats, reason);
}

ParetoFrontier* topdown_mdd_cuda_enumerate(MDD* mdd,
                                           EnumerationStats* stats,
                                           std::string* reason) {
    if (!prepare_cuda_device(reason)) {
        return NULL;
    }
    return enumerate_mdd_topdown(mdd, stats, reason);
}

ParetoFrontier* coupled_cuda_enumerate(MDD* mdd,
                                       EnumerationStats* stats,
                                       std::string* reason) {
    if (!prepare_cuda_device(reason)) {
        return NULL;
    }
    return enumerate_mdd_coupled(mdd, stats, reason);
}
