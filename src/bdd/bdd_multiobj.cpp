// ----------------------------------------------------------
// BDD Multiobjective Algorithms - Implementation
// ----------------------------------------------------------

#include "bdd_multiobj.hpp"
#include "bdd_alg.hpp"
#include <chrono>
#ifdef USE_CUDA
#include "../cuda/topdown_cuda.hpp"
#include "../cuda/coupled_cuda.hpp"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#pragma weak topdown_cuda_enumerate
#pragma weak topdown_mdd_cuda_enumerate
#pragma weak coupled_cuda_enumerate
#endif

typedef std::pair<int,int> intpair;

inline int normalized_cpu_threads(const int cpu_threads) {
    return std::max(1, cpu_threads);
}

inline bool use_parallel_cpu(const int cpu_threads) {
#ifdef _OPENMP
    return normalized_cpu_threads(cpu_threads) > 1;
#else
    (void)cpu_threads;
    return false;
#endif
}

inline ParetoFrontier* request_frontier(ParetoFrontierManager* mgmr, const bool parallel_mode) {
    return parallel_mode ? new ParetoFrontier : mgmr->request();
}

inline void recycle_frontier(ParetoFrontierManager* mgmr, ParetoFrontier* frontier, const bool parallel_mode) {
    if (frontier == NULL) {
        return;
    }
    if (parallel_mode) {
        delete frontier;
    } else {
        mgmr->deallocate(frontier);
    }
}

inline bool IntPairLargestToSmallestComp(intpair l, intpair r) {
    return l.second > r.second;     // from largest to smallest
}

inline bool SetPackingStateMinElementSmallestToLargestComp(Node* l, Node* r) {
    return l->setpack_state.find_first() < r->setpack_state.find_first();     // from smallest to largest
}

typedef std::chrono::steady_clock WallClock;

inline bool cpu_perf_enabled(const MultiObjectiveStats* stats) {
    return (stats != NULL && stats->cpu_perf_enabled);
}

inline double wall_elapsed_s(const WallClock::time_point& start) {
    return std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - start).count();
}

inline void reset_cpu_perf_stats(MultiObjectiveStats* stats) {
    if (stats == NULL) {
        return;
    }
    stats->cpu_expand_td_wall_s = 0.0;
    stats->cpu_expand_bu_wall_s = 0.0;
    stats->cpu_recompute_td_wall_s = 0.0;
    stats->cpu_recompute_bu_wall_s = 0.0;
    stats->cpu_dominance_wall_s = 0.0;
    stats->cpu_cutset_sort_wall_s = 0.0;
    stats->cpu_cutset_convolution_wall_s = 0.0;
    stats->cpu_cutset_partial_merge_wall_s = 0.0;
    stats->cpu_layers_td = 0;
    stats->cpu_layers_bu = 0;
    stats->cpu_nodes_expanded = 0;
    stats->cpu_cutset_size = 0;
}

//
// Find pareto frontier using top-down approach on CUDA
// kernel_version: 1=one-block-per-node, 2=fixed-2D-grid, 3=dynamic-1D-grid
//
ParetoFrontier* BDDMultiObj::pareto_frontier_topdown_cuda(BDD* bdd, bool maximization, const int problem_type, const int dominance_strategy, MultiObjectiveStats* stats, std::string* reason, int kernel_version) {
    if (stats != NULL) {
        stats->pareto_dominance_time = 0;
        stats->pareto_dominance_filtered = 0;
        stats->layer_coupling = 0;
    }

#ifdef USE_CUDA
    if (topdown_cuda_enumerate == NULL) {
        if (reason != NULL) {
            *reason = "CUDA top-down enumeration symbol is unavailable in this binary";
        }
        return NULL;
    }

    std::string local_reason;
    std::string* active_reason = reason != NULL ? reason : &local_reason;
    ParetoFrontier* frontier = topdown_cuda_enumerate(bdd, maximization, problem_type, dominance_strategy, stats, active_reason, kernel_version);
    if (frontier == NULL && reason != NULL && reason->empty()) {
        *reason = "CUDA top-down enumeration failed";
    }
    return frontier;
#else
    (void)bdd;
    (void)maximization;
    (void)problem_type;
    (void)dominance_strategy;
    (void)kernel_version;
    if (reason != NULL) {
        *reason = "GPU backend requested but binary was built without CUDA support";
    }
    return NULL;
#endif
}

//
// Find pareto frontier using top-down approach on CUDA for MDD
// kernel_version: 1=one-block-per-node, 2=fixed-2D-grid, 3=dynamic-1D-grid
//
ParetoFrontier* BDDMultiObj::pareto_frontier_topdown_cuda(MDD* mdd, MultiObjectiveStats* stats, std::string* reason, int kernel_version) {
    if (stats != NULL) {
        stats->pareto_dominance_time = 0;
        stats->pareto_dominance_filtered = 0;
        stats->layer_coupling = 0;
    }

#ifdef USE_CUDA
    std::string local_reason;
    std::string* active_reason = reason != NULL ? reason : &local_reason;
    ParetoFrontier* frontier = topdown_mdd_cuda_enumerate(mdd, stats, active_reason, kernel_version);
    if (frontier == NULL && reason != NULL && reason->empty()) {
        *reason = "CUDA top-down enumeration failed for MDD";
    }
    return frontier;
#else
    (void)mdd;
    (void)kernel_version;
    if (reason != NULL) {
        *reason = "GPU backend requested but binary was built without CUDA support";
    }
    return NULL;
#endif
}

//
// Find pareto frontier using top-down approach
//
ParetoFrontier* BDDMultiObj::pareto_frontier_topdown(BDD* bdd, bool maximization, const int problem_type, int dominance_strategy, MultiObjectiveStats* stats, int cpu_threads) {
    //cout << "\nComputing Pareto Frontier..." << endl;

	// Initialize stats
	stats->pareto_dominance_time = 0;
	stats->pareto_dominance_filtered = 0;
    reset_cpu_perf_stats(stats);
    clock_t init;
    const bool perf_enabled = cpu_perf_enabled(stats);
	
	// Initialize manager
	ParetoFrontierManager* mgmr = new ParetoFrontierManager(bdd->get_width());
    const int threads = normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = use_parallel_cpu(threads);

	// root node
    ObjType zero_array[NOBJS];
	memset(zero_array, 0, sizeof(ObjType)*NOBJS);
	bdd->get_root()->pareto_frontier = request_frontier(mgmr, parallel_mode);
	bdd->get_root()->pareto_frontier->add(zero_array);

    
	if (maximization) {
		for (int l = 1; l < bdd->num_layers; ++l) {
//			cout << "\tLayer " << l << " - size = " << bdd->layers[l].size() << '\n';
            const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
		
            const int layer_size = bdd->layers[l].size();
#ifdef _OPENMP
#pragma omp parallel for if(parallel_mode) num_threads(threads) schedule(dynamic)
#endif
            for (int i = 0; i < layer_size; ++i) {
                Node* node = bdd->layers[l][i];
                node->pareto_frontier = request_frontier(mgmr, parallel_mode);
                for (vector<Node*>::iterator prev = node->prev[1].begin();
                        prev != node->prev[1].end(); ++prev) {
                    node->pareto_frontier->merge(*((*prev)->pareto_frontier), (*prev)->weights[1]);
                }
                for (vector<Node*>::iterator prev = node->prev[0].begin();
                        prev != node->prev[0].end(); ++prev) {
                    node->pareto_frontier->merge(*((*prev)->pareto_frontier), (*prev)->weights[0]);
                }
            }

				if (dominance_strategy > 0) {
                    const WallClock::time_point dominance_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
                    init = clock();
					BDDMultiObj::filter_dominance(bdd, l, problem_type, dominance_strategy, stats);
                    stats->pareto_dominance_time += clock()-init;
                    if (perf_enabled) {
                        stats->cpu_dominance_wall_s += wall_elapsed_s(dominance_begin);
                    }
				}
				
			// Deallocate frontier from previous layer
				for (vector<Node*>::iterator it = bdd->layers[l-1].begin(); it != bdd->layers[l-1].end(); ++it) {
					recycle_frontier(mgmr, (*it)->pareto_frontier, parallel_mode);
				}
                if (perf_enabled) {
                    stats->cpu_expand_td_wall_s += wall_elapsed_s(expand_begin);
                    stats->cpu_layers_td += 1;
                    stats->cpu_nodes_expanded += layer_size;
                }
			}
		} else {
			for (int l = 1; l < bdd->num_layers; ++l) {
				//cout << "\tLayer " << l << " - size = " << bdd->layers[l].size() << '\n';
                const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
		
            const int layer_size = bdd->layers[l].size();
#ifdef _OPENMP
#pragma omp parallel for if(parallel_mode) num_threads(threads) schedule(dynamic)
#endif
            for (int i = 0; i < layer_size; ++i) {
                Node* node = bdd->layers[l][i];
                node->pareto_frontier = request_frontier(mgmr, parallel_mode);
                for (vector<Node*>::iterator prev = node->prev[0].begin();
                        prev != node->prev[0].end(); ++prev) {
                    node->pareto_frontier->merge(*((*prev)->pareto_frontier), (*prev)->weights[0]);
                }
                for (vector<Node*>::iterator prev = node->prev[1].begin();
                        prev != node->prev[1].end(); ++prev) {
                    node->pareto_frontier->merge(*((*prev)->pareto_frontier), (*prev)->weights[1]);
                }
            }

				if (dominance_strategy > 0) {				
                    const WallClock::time_point dominance_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
                    init = clock();				
					BDDMultiObj::filter_dominance(bdd, l, problem_type, dominance_strategy, stats);
					stats->pareto_dominance_time += clock()-init;
                    if (perf_enabled) {
                        stats->cpu_dominance_wall_s += wall_elapsed_s(dominance_begin);
                    }
				}

			// Deallocate frontier from previous layer
				for (vector<Node*>::iterator it = bdd->layers[l-1].begin(); it != bdd->layers[l-1].end(); ++it) {
					recycle_frontier(mgmr, (*it)->pareto_frontier, parallel_mode);
				}
                if (perf_enabled) {
                    stats->cpu_expand_td_wall_s += wall_elapsed_s(expand_begin);
                    stats->cpu_layers_td += 1;
                    stats->cpu_nodes_expanded += layer_size;
                }
			}		
		}
	    
    ParetoFrontier* frontier = bdd->get_terminal()->pareto_frontier;
	    
    // Erase memory
	delete mgmr;
	return frontier;
}



//
// Find pareto frontier using bottom-up approach
//
ParetoFrontier* BDDMultiObj::pareto_frontier_bottomup(BDD* bdd, bool maximization, const int problem_type, const int dominance_strategy, MultiObjectiveStats* stats, int cpu_threads) {
    //cout << "\nComputing Pareto Set...\n";
	(void)problem_type;
	(void)dominance_strategy;
	reset_cpu_perf_stats(stats);
    const bool perf_enabled = cpu_perf_enabled(stats);
    const int threads = normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = use_parallel_cpu(threads);

    // Create pareto frontier manager
	ParetoFrontierManager* mgmr = new ParetoFrontierManager(bdd->get_width());

	// root node
    ObjType zero_array[NOBJS];
	memset(zero_array, 0, sizeof(ObjType)*NOBJS);
	bdd->get_terminal()->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);
	bdd->get_terminal()->pareto_frontier_bu->add(zero_array);

	if (maximization) {
		for (int l = bdd->num_layers-2; l >= 0; --l) {
			// cout << "\tLayer " << l << " - size = " << bdd->layers[l].size();
			// cout << '\n';
            const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();

            const int layer_size = bdd->layers[l].size();
#ifdef _OPENMP
#pragma omp parallel for if(parallel_mode) num_threads(threads) schedule(dynamic)
#endif
            for (int i = 0; i < layer_size; ++i) {
                Node* node = bdd->layers[l][i];
                node->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);
                if (node->arcs[1] != NULL) {
                    node->pareto_frontier_bu->merge(*(node->arcs[1]->pareto_frontier_bu), node->weights[1]);
                }
                if (node->arcs[0] != NULL) {
                    node->pareto_frontier_bu->merge(*(node->arcs[0]->pareto_frontier_bu), node->weights[0]);
                }
            }

			// Deallocate frontier from previous layer
				for (vector<Node*>::iterator it = bdd->layers[l+1].begin(); it != bdd->layers[l+1].end(); ++it) {
					recycle_frontier(mgmr, (*it)->pareto_frontier_bu, parallel_mode);
				}
                if (perf_enabled) {
                    stats->cpu_expand_bu_wall_s += wall_elapsed_s(expand_begin);
                    stats->cpu_layers_bu += 1;
                    stats->cpu_nodes_expanded += layer_size;
                }
			} 
		} else {
			for (int l = bdd->num_layers-2; l >= 0; --l) {
				// cout << "\tLayer " << l << " - size = " << bdd->layers[l].size();
				// cout << '\n';
                const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();

            const int layer_size = bdd->layers[l].size();
#ifdef _OPENMP
#pragma omp parallel for if(parallel_mode) num_threads(threads) schedule(dynamic)
#endif
            for (int i = 0; i < layer_size; ++i) {
                Node* node = bdd->layers[l][i];
                node->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);
                if (node->arcs[0] != NULL) {
                    node->pareto_frontier_bu->merge(*(node->arcs[0]->pareto_frontier_bu), node->weights[0]);
                }
                if (node->arcs[1] != NULL) {
                    node->pareto_frontier_bu->merge(*(node->arcs[1]->pareto_frontier_bu), node->weights[1]);
                }
            }

			// Deallocate frontier from previous layer
				for (vector<Node*>::iterator it = bdd->layers[l+1].begin(); it != bdd->layers[l+1].end(); ++it) {
					recycle_frontier(mgmr, (*it)->pareto_frontier_bu, parallel_mode);
				}
                if (perf_enabled) {
                    stats->cpu_expand_bu_wall_s += wall_elapsed_s(expand_begin);
                    stats->cpu_layers_bu += 1;
                    stats->cpu_nodes_expanded += layer_size;
                }
			} 
		}

    ParetoFrontier* frontier = bdd->get_root()->pareto_frontier_bu;

    // Erase memory
	delete mgmr;

    // Return pareto frontier
	return frontier;
}


//
// Expand pareto frontier / topdown version
//
inline void expand_layer_topdown(BDD* bdd, const int l, const bool maximization, ParetoFrontierManager* mgmr, const int cpu_threads) {
    const int threads = normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = use_parallel_cpu(threads);
	if (maximization) {
        const int layer_size = bdd->layers[l].size();
#ifdef _OPENMP
#pragma omp parallel for if(parallel_mode) num_threads(threads) schedule(dynamic)
#endif
		for (int i = 0; i < layer_size; ++i) {
            Node* node = bdd->layers[l][i];
			// Request frontier
			node->pareto_frontier = request_frontier(mgmr, parallel_mode);

			// add incoming one arcs
			for (vector<Node*>::iterator prev = node->prev[1].begin(); prev != node->prev[1].end(); ++prev) {
				node->pareto_frontier->merge(*((*prev)->pareto_frontier), (*prev)->weights[1]);
			}

			// add incoming zero arcs
			for (vector<Node*>::iterator prev = node->prev[0].begin(); prev != node->prev[0].end(); ++prev) {
				node->pareto_frontier->merge(*((*prev)->pareto_frontier), (*prev)->weights[0]);
			}
		}
	} else {
        const int layer_size = bdd->layers[l].size();
#ifdef _OPENMP
#pragma omp parallel for if(parallel_mode) num_threads(threads) schedule(dynamic)
#endif
		for (int i = 0; i < layer_size; ++i) {
            Node* node = bdd->layers[l][i];
			// Request frontier
			node->pareto_frontier = request_frontier(mgmr, parallel_mode);

			// add incoming zero arcs
			for (vector<Node*>::iterator prev = node->prev[0].begin(); prev != node->prev[0].end(); ++prev) {
				node->pareto_frontier->merge(*((*prev)->pareto_frontier), (*prev)->weights[0]);
			}

			// add incoming one arcs
			for (vector<Node*>::iterator prev = node->prev[1].begin(); prev != node->prev[1].end(); ++prev) {
				node->pareto_frontier->merge(*((*prev)->pareto_frontier), (*prev)->weights[1]);
			}
		}		
	}

	//BDDMultiObj::filter_dominance_knapsack(bdd, l);
	BDDMultiObj::filter_completion(bdd, l);
	
	// deallocate previous layer
	for (int i = 0; i < bdd->layers[l-1].size(); ++i) {
		recycle_frontier(mgmr, bdd->layers[l-1][i]->pareto_frontier, parallel_mode);
	}
}


//
// Expand pareto frontier / bottomup version
//
inline void expand_layer_bottomup(BDD* bdd, const int l, const bool maximization, ParetoFrontierManager* mgmr, const int cpu_threads) {
    const int threads = normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = use_parallel_cpu(threads);
	if (maximization) {
        const int layer_size = bdd->layers[l].size();
#ifdef _OPENMP
#pragma omp parallel for if(parallel_mode) num_threads(threads) schedule(dynamic)
#endif
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
#ifdef _OPENMP
#pragma omp parallel for if(parallel_mode) num_threads(threads) schedule(dynamic)
#endif
		for (int i = 0; i < layer_size; ++i) {
            Node* node = bdd->layers[l][i];

			// Request frontier
			node->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);

			// add outgoing zero arcs
			if (node->arcs[0] != NULL) {
				node->pareto_frontier_bu->merge(*(node->arcs[0]->pareto_frontier_bu),node->weights[0]);
			}

			// add outgoing one arcs
			if (node->arcs[1] != NULL) {
				node->pareto_frontier_bu->merge(*(node->arcs[1]->pareto_frontier_bu), node->weights[1]);
			}
		}		
	}
	// deallocate next layer
	for (int i = 0; i < bdd->layers[l+1].size(); ++i) {
		recycle_frontier(mgmr, bdd->layers[l+1][i]->pareto_frontier_bu, parallel_mode);
	}
}


//
// Expand pareto frontier / topdown version
//
inline void expand_layer_topdown(MDD* mdd, const int l, ParetoFrontierManager* mgmr, const int cpu_threads) {
    const int threads = normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = use_parallel_cpu(threads);
    const int layer_size = mdd->layers[l].size();
#ifdef _OPENMP
#pragma omp parallel for if(parallel_mode) num_threads(threads) schedule(dynamic)
#endif
	for (int i = 0; i < layer_size; ++i) {
        MDDNode* node = mdd->layers[l][i];
		// Request frontier
		node->pareto_frontier = request_frontier(mgmr, parallel_mode);

		// add incoming one arcs
		for (MDDArc* arc : node->in_arcs_list) {
			node->pareto_frontier->merge(*(arc->tail->pareto_frontier), arc->weights);
		}
	}		

	// deallocate previous layer
	for (int i = 0; i < mdd->layers[l-1].size(); ++i) {
		recycle_frontier(mgmr, mdd->layers[l-1][i]->pareto_frontier, parallel_mode);
		delete mdd->layers[l-1][i];
	}
}


//
// Expand pareto frontier / topdown version
//
inline void expand_layer_bottomup(MDD* mdd, const int l, ParetoFrontierManager* mgmr, const int cpu_threads) {
    const int threads = normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = use_parallel_cpu(threads);
    const int layer_size = mdd->layers[l].size();
#ifdef _OPENMP
#pragma omp parallel for if(parallel_mode) num_threads(threads) schedule(dynamic)
#endif
	for (int i = 0; i < layer_size; ++i) {
        MDDNode* node = mdd->layers[l][i];
		// Request frontier
		node->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);

		// add incoming one arcs
		for (MDDArc* arc : node->out_arcs_list) {
			node->pareto_frontier_bu->merge(*(arc->head->pareto_frontier_bu), arc->weights);
		}
	}		

	// deallocate next layer
	for (int i = 0; i < mdd->layers[l+1].size(); ++i) {
		recycle_frontier(mgmr, mdd->layers[l+1][i]->pareto_frontier_bu, parallel_mode);
		delete mdd->layers[l+1][i];
	}
}


//
// Topdown value of a node (for dynamic layer selection)
//
inline int topdown_layer_value(BDD* bdd, Node* node) {
	int total = 0;
	for (int t = 0; t < 2; ++t) {
		if (node->arcs[t] != NULL) {
			total += node->pareto_frontier->get_num_sols();
		}
	}
	return total;
}



//
// Bottomup value of a node (for dynamic layer selection)
//
inline int bottomup_layer_value(BDD* bdd, Node* node) {
	int total = 0;
	for (int t = 0; t < 2; ++t) {
		total += node->pareto_frontier_bu->get_num_sols() * node->prev[t].size();
	}
	return 1.5*total;
}


//
// Topdown value of a node (for dynamic layer selection)
//
inline int topdown_layer_value(MDD* mdd, MDDNode* node) {
	// int total = 0;
	// for (MDDArc* arc : node->out_arcs_list) {
	// 	total += node->pareto_frontier->get_num_sols();
	// }
	// return total;
	return node->pareto_frontier->get_num_sols() * node->out_arcs_list.size();
}

//
// Bottomup value of a node (for dynamic layer selection)
//
inline int bottomup_layer_value(MDD* mdd, MDDNode* node) {
	// int total = 0;
	// for (MDDArc* arc : node->out_arcs_list) {
	// 	total += node->pareto_frontier_bu->get_num_sols();
	// }
	// return 1.5*total;
	return 1.5 * node->pareto_frontier_bu->get_num_sols() * node->in_arcs_list.size();
}


//
// Comparator for node selection in convolution
//
struct CompareNode {
	bool operator()(const Node* nodeA, const Node* nodeB) {
		return (nodeA->pareto_frontier->get_sum() + nodeA->pareto_frontier_bu->get_sum()) > (nodeB->pareto_frontier->get_sum() + nodeB->pareto_frontier_bu->get_sum());
	}
};

//
// Comparator for node selection in convolution
//
struct CompareMDDNode {
	bool operator()(const MDDNode* nodeA, const MDDNode* nodeB) {
		return (nodeA->pareto_frontier->get_sum() + nodeA->pareto_frontier_bu->get_sum()) > (nodeB->pareto_frontier->get_sum() + nodeB->pareto_frontier_bu->get_sum());
	}
};


//
// Find pareto frontier using dynamic layer cutset
//
ParetoFrontier* BDDMultiObj::pareto_frontier_dynamic_layer_cutset(BDD* bdd, bool maximization, const int problem_type, const int dominance_strategy, MultiObjectiveStats* stats, int cpu_threads) {
	// Create pareto frontier manager
	ParetoFrontierManager* mgmr = new ParetoFrontierManager(bdd->get_width());
    const int threads = normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = use_parallel_cpu(threads);
    reset_cpu_perf_stats(stats);
    const bool perf_enabled = cpu_perf_enabled(stats);

	// Create root and terminal frontiers
	ObjType sol[NOBJS];
	memset(sol, 0, sizeof(ObjType)*NOBJS);

	bdd->get_root()->pareto_frontier = request_frontier(mgmr, parallel_mode);
	bdd->get_root()->pareto_frontier->add(sol);

	bdd->get_terminal()->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);
	bdd->get_terminal()->pareto_frontier_bu->add(sol);

	// Initialize stats
	stats->pareto_dominance_time = 0;
	stats->pareto_dominance_filtered = 0;
    clock_t init;

	// Current layers
	int layer_topdown = 0;
	int layer_bottomup = bdd->num_layers-1;

	// Value of layer
	int val_topdown = 0;
	int val_bottomup = 0;

	while (layer_topdown != layer_bottomup) {
//		if (layer_topdown <= 3) {
		if (val_topdown <= val_bottomup) {
            // Expand topdown
            const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
			expand_layer_topdown(bdd, ++layer_topdown, maximization, mgmr, threads);
            if (perf_enabled) {
                stats->cpu_expand_td_wall_s += wall_elapsed_s(expand_begin);
                stats->cpu_layers_td += 1;
                stats->cpu_nodes_expanded += bdd->layers[layer_topdown].size();
            }
			// Recompute layer value
			val_topdown = 0;
            const int layer_size = bdd->layers[layer_topdown].size();
            const WallClock::time_point recompute_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
#ifdef _OPENMP
#pragma omp parallel for if(parallel_mode) num_threads(threads) reduction(+:val_topdown)
#endif
			for (int i = 0; i < layer_size; ++i) {
				val_topdown += topdown_layer_value(bdd, bdd->layers[layer_topdown][i]);
			}
            if (perf_enabled) {
                stats->cpu_recompute_td_wall_s += wall_elapsed_s(recompute_begin);
            }
			//cout << "DOMINANCE: " << dominance_strategy << endl;
            if (dominance_strategy > 0) {
                const WallClock::time_point dominance_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
                init = clock();
				BDDMultiObj::filter_dominance(bdd, layer_topdown, problem_type, dominance_strategy, stats);
                stats->pareto_dominance_time += clock()-init;
                if (perf_enabled) {
                    stats->cpu_dominance_wall_s += wall_elapsed_s(dominance_begin);
                }
			}
		} else {
			// Expand layer bottomup
            const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
			expand_layer_bottomup(bdd, --layer_bottomup, maximization, mgmr, threads);
            if (perf_enabled) {
                stats->cpu_expand_bu_wall_s += wall_elapsed_s(expand_begin);
                stats->cpu_layers_bu += 1;
                stats->cpu_nodes_expanded += bdd->layers[layer_bottomup].size();
            }
			// Recompute layer value
			val_bottomup = 0;
            const int layer_size = bdd->layers[layer_bottomup].size();
            const WallClock::time_point recompute_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
#ifdef _OPENMP
#pragma omp parallel for if(parallel_mode) num_threads(threads) reduction(+:val_bottomup)
#endif
			for (int i = 0; i < layer_size; ++i) {
				val_bottomup += bottomup_layer_value(bdd, bdd->layers[layer_bottomup][i]);
			}
            if (perf_enabled) {
                stats->cpu_recompute_bu_wall_s += wall_elapsed_s(recompute_begin);
            }
		}

		// if (layer_topdown != old_topdown && (layer_bottomup - layer_topdown <= 3)) {
		// 	cout << "\nFiltering..." << endl;				
		// 	old_topdown = layer_topdown;
		// 	BDDMultiObj::filter_dominance_knapsack(bdd, layer_topdown);
		// } 
		
		//cout << "\tTD=" << layer_topdown << "\tBU=" << layer_bottomup << "\tV-TD=" << val_topdown << "\tV-BU=" << val_bottomup << endl;
	}

	// Save stats
	stats->layer_coupling = layer_topdown;

	// Coupling
	//cout << "\nCoupling..." << endl;

	vector<Node*>& cutset = bdd->layers[layer_topdown];	
	//cout << "\tCutset size: " << cutset.size() << endl;
    if (perf_enabled) {
        stats->cpu_cutset_size = cutset.size();
    }

	//cout << "\tsorting..." << endl;
    const WallClock::time_point cutset_sort_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
	sort(cutset.begin(), cutset.end(), CompareNode());
    if (perf_enabled) {
        stats->cpu_cutset_sort_wall_s += wall_elapsed_s(cutset_sort_begin);
    }

	// Compute expected frontier size
	long int expected_size = 0;
	for (int i = 0; i < cutset.size(); ++i) {
		expected_size += cutset[i]->pareto_frontier->get_num_sols() 
				* cutset[i]->pareto_frontier_bu->get_num_sols(); 
	}
	expected_size = 10000;

	ParetoFrontier* paretoFrontier = new ParetoFrontier;
	paretoFrontier->sols.reserve( expected_size * NOBJS );

    if (parallel_mode && cutset.size() > 1) {
#ifdef _OPENMP
        const WallClock::time_point convolution_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        vector<ParetoFrontier*> partial(threads, NULL);
#pragma omp parallel num_threads(threads)
        {
            const int tid = omp_get_thread_num();
            ParetoFrontier* local_frontier = new ParetoFrontier;
            partial[tid] = local_frontier;
#pragma omp for schedule(dynamic)
            for (int i = 0; i < cutset.size(); ++i) {
                Node* node = cutset[i];
                assert( node->pareto_frontier != NULL );
                assert( node->pareto_frontier_bu != NULL );
                local_frontier->convolute( *(node->pareto_frontier), *(node->pareto_frontier_bu) );
            }
        }
        if (perf_enabled) {
            stats->cpu_cutset_convolution_wall_s += wall_elapsed_s(convolution_begin);
        }
        const WallClock::time_point partial_merge_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        for (int t = 0; t < threads; ++t) {
            if (partial[t] != NULL) {
                paretoFrontier->merge(*partial[t]);
                delete partial[t];
            }
        }
        if (perf_enabled) {
            stats->cpu_cutset_partial_merge_wall_s += wall_elapsed_s(partial_merge_begin);
        }
#else
        const WallClock::time_point convolution_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        for (int i = 0; i < cutset.size(); ++i) {
            Node* node = cutset[i];
            assert( node->pareto_frontier != NULL );
            assert( node->pareto_frontier_bu != NULL );
            paretoFrontier->convolute( *(node->pareto_frontier), *(node->pareto_frontier_bu) );
        }
        if (perf_enabled) {
            stats->cpu_cutset_convolution_wall_s += wall_elapsed_s(convolution_begin);
        }
#endif
    } else {
        const WallClock::time_point convolution_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        for (int i = 0; i < cutset.size(); ++i) {
            Node* node = cutset[i];
            assert( node->pareto_frontier != NULL );
            assert( node->pareto_frontier_bu != NULL );
            paretoFrontier->convolute( *(node->pareto_frontier), *(node->pareto_frontier_bu) );
        }
        if (perf_enabled) {
            stats->cpu_cutset_convolution_wall_s += wall_elapsed_s(convolution_begin);
        }
    }
    
    // deallocate manager
	delete mgmr;

    // return pareto frontier
	return paretoFrontier;
}


//
// S-T Relaxation Policies 
//
struct S_T_Policies {
    // Index array
    vector<int> indices;
    // Comparator metric
    vector<ObjType> metric;
    // Auxiliary solution type
    vector<ObjType> sols_aux;
	// Max size of T set
	const int t_max;
	// Max size of S set
	const int s_max;
	// Auxiliary image point
	ObjType* aux_point;

    // Solution comparator
    struct SolComparator {
        vector<ObjType>& metric;
        // Constructor
        SolComparator(vector<ObjType>& _metric) : metric(_metric) 
        { }
        // Compare based on metric
        bool operator()(const int indexA, const int indexB) {
            return metric[indexA] > metric[indexB];
        } 
    };
    SolComparator sol_comparator;

    //
    // Constructor
    //
    S_T_Policies(const int _t_max, const int _s_max) 
		: sol_comparator(SolComparator(metric)), t_max(_t_max), s_max(_s_max)
    { 
		aux_point = new ObjType[NOBJS];
	}

	//
	// Destructor
	//
	~S_T_Policies() {
		delete[] aux_point;
	}

    //
    // Relax T set
    //
    void relax_T(ParetoFrontier* pf_T, ParetoFrontier* pf_S) {

		// cout << "Set T Original: " << endl;
		// pf_T->print();

        // Compute metric for comparison
        const int num_sols_T = pf_T->get_num_sols();
        indices.resize(num_sols_T);
        metric.resize(num_sols_T);        
        for (int i = 0; i < num_sols_T; ++i) {
            indices[i] = i;
            metric[i] = 0.0;
            for (int k = i * NOBJS; k < i * NOBJS + NOBJS; ++k) {
                metric[i] += pf_T->sols[k];
            }
        }

        // Sort elements based on metric
        sort(indices.begin(), indices.end(), sol_comparator);
        sols_aux = pf_T->sols;
        for (int i = 0; i < num_sols_T; ++i) {
            for (int p = 0; p < NOBJS; ++p) {
                pf_T->sols[i * NOBJS + p] = sols_aux[ indices[i] * NOBJS + p ];
            }
        }

		//cout << "t_max = " << t_max << " - size: " << num_sols_T << endl;

        // Transfer last elements to S set
		for (int i = t_max; i < num_sols_T; ++i) {
			// Add to S set
			pf_S->add( &(pf_T->sols[i*NOBJS]) );
			// Remove from set
			pf_T->sols[i*NOBJS] = DOMINATED;
		}
		
		// Clean T set
		pf_T->remove_dominated();

		// cout << "\nSet T: " << endl;
		// pf_T->print();
		// cout << "Set S: " << endl;
		// pf_S->print();
    }


    //
    // Relax S set
    //
    void relax_S(ParetoFrontier* pf_T, ParetoFrontier* pf_S) {

		// cout << "Set S Original: " << endl;
		// pf_S->print();

        // Compute metric for comparison
        const int num_sols_S = pf_S->get_num_sols();
        indices.resize(num_sols_S);
        metric.resize(num_sols_S);        
        for (int i = 0; i < num_sols_S; ++i) {
            indices[i] = i;
            metric[i] = 0.0;
            for (int k = i * NOBJS; k < i * NOBJS + NOBJS; ++k) {
                metric[i] += pf_S->sols[k];
            }
        }

        // Sort elements based on metric
        sort(indices.begin(), indices.end(), sol_comparator);
        sols_aux = pf_S->sols;
        for (int i = 0; i < num_sols_S; ++i) {
            for (int p = 0; p < NOBJS; ++p) {
                pf_S->sols[i * NOBJS + p] = sols_aux[ indices[i] * NOBJS + p ];
            }
        }

		// Compute relaxed dominated point

		// assert( pf_S->sols.size() < (s_max-1)*NOBJS );
		// assert( pf_S->sols.size() < (s_max-1)*NOBJS );

		std::copy(pf_S->sols.begin()+(s_max-1)*NOBJS, pf_S->sols.begin()+(s_max)*NOBJS, aux_point);


		pf_S->sols[(s_max-1)*NOBJS] = DOMINATED;
		for (int i = s_max; i < num_sols_S; ++i) {
			// Compute ideal completion with respect to two points
			for (int p = 0; p < NOBJS; ++p) {
				aux_point[p] = std::max(aux_point[p], pf_S->sols[i*NOBJS+p]);
			}
			// Eliminate point from S
			pf_S->sols[i*NOBJS] = DOMINATED;
		}
		// Clean S set
		pf_S->remove_dominated();
		pf_S->add(aux_point);

		// //cout << "t_max = " << t_max << " - size: " << num_sols_T << endl;

        // // Transfer last elements to S set
		// for (int i = t_max; i < num_sols_T; ++i) {
		// 	// Add to S set
		// 	pf_S->add( &(pf_T->sols[i*NOBJS]) );
		// 	// Remove from set
		// 	pf_T->sols[i*NOBJS] = DOMINATED;
		// }
		
		// // Clean T set
		// pf_T->remove_dominated();

		// cout << endl;
		// cout << "Set S - final: " << endl;
		// pf_S->print();
		// exit(1);
    }


};


//
// Approximate pareto frontier / top-down
//
void BDDMultiObj::approximate_pareto_frontier_topdown_dominance(BDD* bdd, const int t_max, const int s_max) {
    cout << "\nComputing approximation..." << endl;

	// Create pareto frontier manager
	ParetoFrontierManager* mgmr = new ParetoFrontierManager(bdd->get_width());

	// S-T Relaxation policies
	S_T_Policies relax_policy(t_max, s_max);

	// initialize root node sets
	ObjType zero_array[NOBJS];
	memset(zero_array, 0, sizeof(ObjType)*NOBJS);

	// S-set contain relaxation - it can be empty
	bdd->get_root()->pareto_frontier_S = mgmr->request();

	// T-set contains feasible solutions - it contains 0 array
	bdd->get_root()->pareto_frontier_T = mgmr->request();
	bdd->get_root()->pareto_frontier_T->add(zero_array);

	// Auxiliary for filtering
	ObjType* aux_point = new ObjType[NOBJS];

	int num_arcs_removed = 0;

	for (int l = 1; l < bdd->num_layers; ++l) {
		cout << "\tLayer " << l << " - size = " << bdd->layers[l].size() << '\n';
	
		// iterate on layers
		for (vector<Node*>::iterator it = bdd->layers[l].begin(); it != bdd->layers[l].end(); ++it) {

			Node* node = (*it);		
			int id = node->index;

			// Compute S and T sets

			// Request frontier
			node->pareto_frontier_T = mgmr->request();
			node->pareto_frontier_S = mgmr->request();

			// add incoming one arcs to both T and S sets
			for (vector<Node*>::iterator prev = (*it)->prev[1].begin();
					prev != (*it)->prev[1].end(); ++prev) {
				node->pareto_frontier_T->merge(*((*prev)->pareto_frontier_T), (*prev)->weights[1]);
				node->pareto_frontier_S->merge(*((*prev)->pareto_frontier_S), (*prev)->weights[1]);                     
			}
			
			// add incoming zero arcs to both T and S sets
			for (vector<Node*>::iterator prev = (*it)->prev[0].begin();
					prev != (*it)->prev[0].end(); ++prev) {
				node->pareto_frontier_T->merge(*((*prev)->pareto_frontier_T));
				node->pareto_frontier_S->merge(*((*prev)->pareto_frontier_S));
			}

			// Remove sets from S that are dominated from T
			int num_dominated = 0;
			for (int i = 0; i < node->pareto_frontier_T->sols.size(); i += NOBJS) {
				for (int j = 0; j < node->pareto_frontier_S->sols.size(); j += NOBJS) {
					bool dominated = true;
					for (int p = 0; p < NOBJS && dominated; ++p) {
						dominated = ( node->pareto_frontier_T->sols[i+p] >= node->pareto_frontier_S->sols[j+p] );
					}
					if (dominated) {
						node->pareto_frontier_S->sols[j] = DOMINATED;
						++num_dominated;
					}
				} 
			}
			if (num_dominated > 0) {
				node->pareto_frontier_S->remove_dominated();
			}

			// relax T-set if maximum cardinality is violated
			if (node->pareto_frontier_T->get_num_sols() > t_max) {
				relax_policy.relax_T(node->pareto_frontier_T, node->pareto_frontier_S);
			}

			// relax S-set if maximum cardinality is violated
			if (node->pareto_frontier_S->get_num_sols() > s_max) {
				relax_policy.relax_S(node->pareto_frontier_T, node->pareto_frontier_S);
			}
		}

		BDDMultiObj::filter_dominance_knapsack_approx(bdd, l);

		// Deallocate frontier from previous layer
		for (vector<Node*>::iterator it = bdd->layers[l-1].begin(); it != bdd->layers[l-1].end(); ++it) {
			mgmr->deallocate( (*it)->pareto_frontier_S );
			mgmr->deallocate( (*it)->pareto_frontier_T );
		}
	}
	bdd->remove_dangling_nodes();
	bdd->fix_indices();

	cout << "\tSize of set T at terminal: ";
	cout << bdd->get_terminal()->pareto_frontier_T->get_num_sols();
	cout << endl;

	cout << "\tSize of set S at terminal: ";
	cout << bdd->get_terminal()->pareto_frontier_S->get_num_sols();
	cout << endl;

	int num_nodes_removed = 0;
	if (num_arcs_removed > 0) {
		num_nodes_removed = bdd->get_num_nodes();
		// remove dangling nodes
		bdd->remove_dangling_nodes();
		bdd->fix_indices();
		num_nodes_removed = num_nodes_removed - bdd->get_num_nodes();
	}

	cout << "\n\nApproximation filtering results: " << endl;
	cout << "\tNumber of arcs removed: " << num_arcs_removed << endl;
	cout << "\tNumber of nodes removed: " << num_nodes_removed << endl;
	cout << "\tMDD width: " << bdd->get_width() << endl;
	cout << "\tMDD nodes: " << bdd->get_num_nodes() << endl;
	cout << endl;


	// cout << "\tMerged set: " << endl;
	// bdd->get_terminal()->pareto_frontier_S->merge( *(bdd->get_terminal()->pareto_frontier_T) );
	// cout << bdd->get_terminal()->pareto_frontier_S->get_num_sols();
	// cout << endl;

	// deallocate manager
	delete mgmr;
	delete[] aux_point;	
}


//
// Approximate pareto frontier / top-down
//
void BDDMultiObj::approximate_pareto_frontier_topdown(BDD* bdd, const int t_max, const int s_max) {
    cout << "\nComputing approximation..." << endl;

	// Create pareto frontier manager
	ParetoFrontierManager* mgmr = new ParetoFrontierManager(bdd->get_width());

    // S-T Relaxation policies
    S_T_Policies relax_policy(t_max, s_max);

	// initialize root node sets
    ObjType zero_array[NOBJS];
	memset(zero_array, 0, sizeof(ObjType)*NOBJS);

    // S-set contain relaxation - it can be empty
	bdd->get_root()->pareto_frontier_S = mgmr->request();

    // T-set contains feasible solutions - it contains 0 array
	bdd->get_root()->pareto_frontier_T = mgmr->request();
	bdd->get_root()->pareto_frontier_T->add(zero_array);

	// Auxiliary for filtering
	ObjType* aux_point = new ObjType[NOBJS];

	int num_arcs_removed = 0;

    for (int l = 1; l < bdd->num_layers; ++l) {
        cout << "\tLayer " << l << " - size = " << bdd->layers[l].size() << '\n';
    
        // iterate on layers
        for (vector<Node*>::iterator it = bdd->layers[l].begin(); it != bdd->layers[l].end(); ++it) {

            Node* node = (*it);		
            int id = node->index;

			// Compute S and T sets

            // Request frontier
            node->pareto_frontier_T = mgmr->request();
            node->pareto_frontier_S = mgmr->request();

            // add incoming one arcs to both T and S sets
            for (vector<Node*>::iterator prev = (*it)->prev[1].begin();
                    prev != (*it)->prev[1].end(); ++prev) {
                node->pareto_frontier_T->merge(*((*prev)->pareto_frontier_T), (*prev)->weights[1]);
                node->pareto_frontier_S->merge(*((*prev)->pareto_frontier_S), (*prev)->weights[1]);                     
            }
            
            // add incoming zero arcs to both T and S sets
            for (vector<Node*>::iterator prev = (*it)->prev[0].begin();
                    prev != (*it)->prev[0].end(); ++prev) {
                node->pareto_frontier_T->merge(*((*prev)->pareto_frontier_T));
                node->pareto_frontier_S->merge(*((*prev)->pareto_frontier_S));
            }

			// Remove sets from S that are dominated from T
			int num_dominated = 0;
			for (int i = 0; i < node->pareto_frontier_T->sols.size(); i += NOBJS) {
				for (int j = 0; j < node->pareto_frontier_S->sols.size(); j += NOBJS) {
					bool dominated = true;
					for (int p = 0; p < NOBJS && dominated; ++p) {
						dominated = ( node->pareto_frontier_T->sols[i+p] >= node->pareto_frontier_S->sols[j+p] );
					}
					if (dominated) {
						node->pareto_frontier_S->sols[j] = DOMINATED;
                        ++num_dominated;
					}
				} 
			}
			if (num_dominated > 0) {
				node->pareto_frontier_S->remove_dominated();
			}

			// *** Filtering ***
			// For each incoming arc, check if the resulting T strictly dominates implied pareto frontier
			for (int arc_type = 0; arc_type < 2; ++arc_type) {
				int arc_index = 0;
				while (arc_index < node->prev[arc_type].size()) {
					// Obtain arc root node
					Node* prev = node->prev[arc_type][arc_index];

					// Check if prev T set is strictly dominated by node T and node Sset
					bool strictly_dominates = true;
					for (int i = 0; i < prev->pareto_frontier_T->sols.size() && strictly_dominates; i += NOBJS) {
						// adjust point
						for (int p = 0; p < NOBJS; ++p) {
							aux_point[p] = prev->weights[arc_type][p] + prev->pareto_frontier_T->sols[i+p];
						}
						for (int j = 0; j < node->pareto_frontier_T->sols.size() && strictly_dominates; j += NOBJS) {
							for (int p = 0; p < NOBJS && strictly_dominates; ++p) {
								strictly_dominates = (aux_point[p] < node->pareto_frontier_T->sols[j+p]);
							}
						}
					}
					// Check if prev S set is strictly dominated by node T and node S set
					for (int i = 0; i < prev->pareto_frontier_S->sols.size() && strictly_dominates; i += NOBJS) {
						// adjust point
						for (int p = 0; p < NOBJS; ++p) {
							aux_point[p] = prev->weights[arc_type][p] + prev->pareto_frontier_S->sols[i+p];
						}
						for (int j = 0; j < node->pareto_frontier_T->sols.size() && strictly_dominates; j += NOBJS) {
							for (int p = 0; p < NOBJS && strictly_dominates; ++p) {
								strictly_dominates = (aux_point[p] < node->pareto_frontier_T->sols[j+p]);
							}
						}
					}
					
					if (strictly_dominates) {
						++num_arcs_removed;
						//cout << "Strictly dominates!" << endl;
						// Remove arc
						prev->arcs[arc_type] = NULL;
						node->prev[arc_type][arc_index] = node->prev[arc_type].back();
						node->prev[arc_type].pop_back();
					} else {
						// Just continue iteration
						++arc_index;
					}
				}
			}

            // relax T-set if maximum cardinality is violated
            if (node->pareto_frontier_T->get_num_sols() > t_max) {
                relax_policy.relax_T(node->pareto_frontier_T, node->pareto_frontier_S);
            }

            // relax S-set if maximum cardinality is violated
            if (node->pareto_frontier_S->get_num_sols() > s_max) {
                relax_policy.relax_S(node->pareto_frontier_T, node->pareto_frontier_S);
            }
        }

		// Deallocate frontier from previous layer
		for (vector<Node*>::iterator it = bdd->layers[l-1].begin(); it != bdd->layers[l-1].end(); ++it) {
			mgmr->deallocate( (*it)->pareto_frontier_S );
			mgmr->deallocate( (*it)->pareto_frontier_T );
		}
    }

	cout << "\tSize of set T at terminal: ";
	cout << bdd->get_terminal()->pareto_frontier_T->get_num_sols();
	cout << endl;

	cout << "\tSize of set S at terminal: ";
	cout << bdd->get_terminal()->pareto_frontier_S->get_num_sols();
	cout << endl;

	int num_nodes_removed = 0;
	if (num_arcs_removed > 0) {
		num_nodes_removed = bdd->get_num_nodes();
		// remove dangling nodes
		bdd->remove_dangling_nodes();
		bdd->fix_indices();
		num_nodes_removed = num_nodes_removed - bdd->get_num_nodes();
	}

	cout << "\n\nApproximation filtering results: " << endl;
	cout << "\tNumber of arcs removed: " << num_arcs_removed << endl;
	cout << "\tNumber of nodes removed: " << num_nodes_removed << endl;
	cout << "\tMDD width: " << bdd->get_width() << endl;
	cout << "\tMDD nodes: " << bdd->get_num_nodes() << endl;
	cout << endl;


	// cout << "\tMerged set: " << endl;
	// bdd->get_terminal()->pareto_frontier_S->merge( *(bdd->get_terminal()->pareto_frontier_T) );
	// cout << bdd->get_terminal()->pareto_frontier_S->get_num_sols();
	// cout << endl;

    // deallocate manager
	delete mgmr;
	delete[] aux_point;
}


//
// Approximate pareto frontier / bottom-up
//
void BDDMultiObj::approximate_pareto_frontier_bottomup(BDD* bdd, const int t_max, const int s_max) {
    cout << "\nComputing approximation..." << endl;

	// Create pareto frontier manager
	ParetoFrontierManager* mgmr = new ParetoFrontierManager(bdd->get_width());

    // S-T Relaxation policies
    S_T_Policies relax_policy(t_max, s_max);

	// initialize root node sets
    ObjType zero_array[NOBJS];
	memset(zero_array, 0, sizeof(ObjType)*NOBJS);

    // S-set contain relaxation - it can be empty
	bdd->get_terminal()->pareto_frontier_S = mgmr->request();

    // T-set contains feasible solutions - it contains 0 array
	bdd->get_terminal()->pareto_frontier_T = mgmr->request();
	bdd->get_terminal()->pareto_frontier_T->add(zero_array);

	// Auxiliary for filtering
	ObjType* aux_point = new ObjType[NOBJS];

	int num_arcs_removed = 0;

    for (int l = bdd->num_layers-2; l >= 0; --l) {
        cout << "\tLayer " << l << " - size = " << bdd->layers[l].size() << '\n';
    
        // iterate on layers
        for (vector<Node*>::iterator it = bdd->layers[l].begin(); it != bdd->layers[l].end(); ++it) {

            Node* node = (*it);		
            int id = node->index;

			// Compute S and T sets

            // Request frontier
            node->pareto_frontier_T = mgmr->request();
            node->pareto_frontier_S = mgmr->request();

			// add outgoing one arcs to both T and S sets
			if (node->arcs[1] != NULL) {
				node->pareto_frontier_T->merge(*(node->arcs[1]->pareto_frontier_T), node->weights[1]);
				node->pareto_frontier_S->merge(*(node->arcs[1]->pareto_frontier_S), node->weights[1]);
			}

            // add outgoing zero arcs to both T and S sets
			if (node->arcs[0] != NULL) {
				node->pareto_frontier_T->merge(*(node->arcs[0]->pareto_frontier_T));
				node->pareto_frontier_S->merge(*(node->arcs[0]->pareto_frontier_S));
			}

			// Remove sets from S that are dominated from T
			int num_dominated = 0;
			for (int i = 0; i < node->pareto_frontier_T->sols.size(); i += NOBJS) {
				for (int j = 0; j < node->pareto_frontier_S->sols.size(); j += NOBJS) {
					bool dominated = true;
					for (int p = 0; p < NOBJS && dominated; ++p) {
						dominated = ( node->pareto_frontier_T->sols[i+p] >= node->pareto_frontier_S->sols[j+p] );
					}
					if (dominated) {
						node->pareto_frontier_S->sols[j] = DOMINATED;
                        ++num_dominated;
					}
				} 
			}
			if (num_dominated > 0) {
				node->pareto_frontier_S->remove_dominated();
			}

			// *** Filtering ***
			// For each incoming arc, check if the resulting T strictly dominates implied pareto frontier
			for (int arc_type = 0; arc_type < 2; ++arc_type) {
				if (node->arcs[arc_type] != NULL) {
					// Obtain arc target node
					Node* target = node->arcs[arc_type];

					// Check if prev T set is strictly dominated by node T and node S set
					bool strictly_dominates = true;
					for (int i = 0; i < target->pareto_frontier_T->sols.size() && strictly_dominates; ++i) {
						for (int j = 0; j < node->pareto_frontier_T->sols.size() && strictly_dominates; ++j) {
							strictly_dominates = 
								(target->pareto_frontier_T->sols[i] + node->weights[arc_type][i % NOBJS] 
								<
								node->pareto_frontier_T->sols[j]);
						}
					}
					for (int i = 0; i < target->pareto_frontier_S->sols.size() && strictly_dominates; ++i) {
						for (int j = 0; j < node->pareto_frontier_T->sols.size() && strictly_dominates; ++j) {
							strictly_dominates = 
								(target->pareto_frontier_S->sols[i] + node->weights[arc_type][i % NOBJS] 
								<
								node->pareto_frontier_T->sols[j]);
						}
					}					
					if (strictly_dominates) {
						++num_arcs_removed;
						//cout << "Strictly dominates!" << endl;
						// Remove arc
						for (int i = 0; i < node->arcs[arc_type]->prev[arc_type].size(); ++i) {
							if (node->arcs[arc_type]->prev[arc_type][i] == node) {
								node->arcs[arc_type]->prev[arc_type][i] = node->arcs[arc_type]->prev[arc_type].back();
								node->arcs[arc_type]->prev[arc_type].pop_back();
								break;
							}
						}
						node->arcs[arc_type] = NULL;
						
						// prev->arcs[arc_type] = NULL;
						// node->prev[arc_type][arc_index] = node->prev[arc_type].back();
						// node->prev[arc_type].pop_back();
					} 
				}
			}

            // relax T-set if maximum cardinality is violated
            if (node->pareto_frontier_T->get_num_sols() > t_max) {
                relax_policy.relax_T(node->pareto_frontier_T, node->pareto_frontier_S);
            }

            // relax S-set if maximum cardinality is violated
            if (node->pareto_frontier_S->get_num_sols() > s_max) {
                relax_policy.relax_S(node->pareto_frontier_T, node->pareto_frontier_S);
            }
        }

		// Deallocate frontier from next layer
		for (vector<Node*>::iterator it = bdd->layers[l+1].begin(); it != bdd->layers[l+1].end(); ++it) {
			mgmr->deallocate( (*it)->pareto_frontier_S );
			mgmr->deallocate( (*it)->pareto_frontier_T );
		}
    }

	cout << "\tSize of set T at root: ";
	cout << bdd->get_root()->pareto_frontier_T->get_num_sols();
	cout << endl;

	cout << "\tSize of set S at root: ";
	cout << bdd->get_root()->pareto_frontier_S->get_num_sols();
	cout << endl;

	int num_nodes_removed = 0;
	if (num_arcs_removed > 0) {
		num_nodes_removed = bdd->get_num_nodes();
		// remove dangling nodes
		bdd->remove_dangling_nodes();
		bdd->fix_indices();
		num_nodes_removed = num_nodes_removed - bdd->get_num_nodes();
	}

	cout << "\n\nApproximation filtering results: " << endl;
	cout << "\tNumber of arcs removed: " << num_arcs_removed << endl;
	cout << "\tNumber of nodes removed: " << num_nodes_removed << endl;
	cout << "\tMDD width: " << bdd->get_width() << endl;
	cout << "\tMDD nodes: " << bdd->get_num_nodes() << endl;
	cout << endl;


	// cout << "\tMerged set: " << endl;
	// bdd->get_terminal()->pareto_frontier_S->merge( *(bdd->get_terminal()->pareto_frontier_T) );
	// cout << bdd->get_terminal()->pareto_frontier_S->get_num_sols();
	// cout << endl;

    // deallocate manager
	delete mgmr;
	delete[] aux_point;
}


//
// Filter layer based on node completion
//
void BDDMultiObj::filter_completion(BDD* bdd, const int layer) {
	//exit(1);
}



//
// Filter layer based on dominance
//
inline void BDDMultiObj::filter_dominance(BDD* bdd, const int layer, const int problem_type, const int dominance_strategy, MultiObjectiveStats* stats) {
	if (problem_type == 1) {
		// Knapsack
		filter_dominance_knapsack(bdd, layer, stats);
	} else if (problem_type == 2) {
		// Set packing
        filter_dominance_setpacking(bdd, layer, stats);
	} else if (problem_type == 3) {
		// Set covering
        filter_dominance_setcovering(bdd, layer, stats);
	}
}



//
// Filter layer based on dominance / knapsack
//
void BDDMultiObj::filter_dominance_knapsack(BDD* bdd, const int layer, MultiObjectiveStats* stats) {
	//	cout << "Applying filter dominance for knapsack..." << endl;
	
	// if (layer > bdd->num_layers/3+10) {
	// 	return;
	// }

    if(bdd->layers[layer].size() > 1) {
        // Compare the nodes based on their min weights, from largets to smallest
        // Ex: 15 <= 11 <= 8 <= 5 <= 0
        vector<intpair> NodeOrder_Weight;
        for(int i=0; i < bdd->layers[layer].size(); i++) {
            if(!bdd->layers[layer][i]->pareto_frontier->sols.empty())
                NodeOrder_Weight.push_back(intpair(i,bdd->layers[layer][i]->min_weight));
        }
        std::sort(NodeOrder_Weight.begin(),NodeOrder_Weight.end(),IntPairLargestToSmallestComp);
        
        // k can be dominated by k+1,k+2,...
        for(int i=0; i < NodeOrder_Weight.size()-1; i++) {
            int index1 = NodeOrder_Weight[i].first;
            Node* node1 = bdd->layers[layer][index1];
            
            // Check each label of node at index1 whether it is dominated or not
            int num_dominated = 0;
            
            //for(int j=i+1; j < NodeOrder_Weight.size(); j++) {
            for(int j=i+1; j < std::min((int)NodeOrder_Weight.size(), i+3); j++) {  // i+3 is just a chosen strategy in order not to try all pairs
                
                int index2 = NodeOrder_Weight[j].first;
                Node* node2 = bdd->layers[layer][index2];
                
                // if node1 and node 2 have one parent and they are the same, continue
                if (node1->prev[0].size() + node1->prev[1].size() == 1
                    && node2->prev[0].size() + node2->prev[1].size() == 1)
                {
                    if (node1->prev[0].size() > 0 && node2->prev[1].size() > 0 && node1->prev[0][0] == node2->prev[1][0])
                        continue;
                    if (node1->prev[1].size() > 0 && node2->prev[0].size() > 0 && node1->prev[1][0] == node2->prev[0][0])
                        continue;	
                }
                
                for (int s1 = 0; s1 < node1->pareto_frontier->sols.size(); s1 += NOBJS) {
                    if(node1->pareto_frontier->sols[s1] == DOMINATED)
                        continue;
                    
                    bool dominated = true;
                    for (int s2 = 0; s2 < node2->pareto_frontier->sols.size(); s2 += NOBJS) {
                        dominated = true;
                        for (int p = 0; p < NOBJS && dominated; ++p)
                            dominated = ( node2->pareto_frontier->sols[s2+p] >= node1->pareto_frontier->sols[s1+p] );
                        if (dominated) {
                            node1->pareto_frontier->sols[s1] = DOMINATED;
                            num_dominated++;
                            break;
                        }
                    }
                }
            }
            
            if (num_dominated > 0) {
				assert (stats != NULL);
				//cout << "\tBefore: " << node1->pareto_frontier->get_num_sols() << endl;
				node1->pareto_frontier->remove_dominated();
				//cout << "\tAfter: " << node1->pareto_frontier->get_num_sols() << endl << endl;				
				stats->pareto_dominance_filtered += num_dominated;
            }
//            cout << "layer: " << layer << ", node: " << node1->index << " , num_dominated: " << num_dominated << endl;
        }
	}
	//cout << "Layer " << layer << " - total filtered: " << total << endl;
}
	

//
// Filter layer based on dominance / knapsack
//
void BDDMultiObj::filter_dominance_knapsack_approx(BDD* bdd, const int layer) {
	//	cout << "Applying filter dominance for knapsack..." << endl;
		
	// if (layer > bdd->num_layers/3+10) {
	// 	return;
	// }

	int total = 0;
	if(bdd->layers[layer].size() > 1) {
		// Compare the nodes based on their min weights, from largets to smallest
		// Ex: 15 <= 11 <= 8 <= 5 <= 0
		vector<intpair> NodeOrder_Weight;
		for(int i=0; i < bdd->layers[layer].size(); i++) {
			//if(!bdd->layers[layer][i]->pareto_frontier_T->sols.empty())
				NodeOrder_Weight.push_back(intpair(i,bdd->layers[layer][i]->min_weight));
		}
		std::sort(NodeOrder_Weight.begin(),NodeOrder_Weight.end(),IntPairLargestToSmallestComp);
		
		// k can be dominated by k+1,k+2,...
		for(int i=0; i < NodeOrder_Weight.size()-1; i++) {
			int index1 = NodeOrder_Weight[i].first;
			Node* node1 = bdd->layers[layer][index1];
			
			// Check each label of node at index1 whether it is dominated or not
			int num_dominated = 0;
			for (int s1 = 0; s1 < node1->pareto_frontier_S->sols.size(); s1 += NOBJS) {
				
				//for(int j=i+1; j < NodeOrder_Weight.size(); j++) {
				for(int j=i+1; j < std::min((int)NodeOrder_Weight.size(), i+3); j++) {
					//int j=i+1;
					int index2 = NodeOrder_Weight[j].first;
					Node* node2 = bdd->layers[layer][index2];
					
					bool dominated = true;
					for (int s2 = 0; s2 < node2->pareto_frontier_T->sols.size(); s2 += NOBJS) {
						dominated = true;
						for (int p = 0; p < NOBJS && dominated; ++p) {
							dominated = ( node2->pareto_frontier_T->sols[s2+p] >= node1->pareto_frontier_S->sols[s1+p] );
						}
						if (dominated) {
							node1->pareto_frontier_S->sols[s1] = DOMINATED;
							num_dominated++;
							break;
						}
					}
					if (dominated)
						break;
				}
			}
			
			if (num_dominated > 0) {
				//cout << "\tBefore: " << node1->pareto_frontier->get_num_sols() << endl;
				node1->pareto_frontier_S->remove_dominated();
				//cout << "\tAfter: " << node1->pareto_frontier->get_num_sols() << endl << endl;				
				total += num_dominated;
			}


			// Check each label of node at index1 whether it is dominated or not
			num_dominated = 0;
			for (int s1 = 0; s1 < node1->pareto_frontier_T->sols.size(); s1 += NOBJS) {
				
				//for(int j=i+1; j < NodeOrder_Weight.size(); j++) {
				for(int j=i+1; j < std::min((int)NodeOrder_Weight.size(), i+3); j++) {
					//int j=i+1;
					int index2 = NodeOrder_Weight[j].first;
					Node* node2 = bdd->layers[layer][index2];
					
					bool dominated = true;
					for (int s2 = 0; s2 < node2->pareto_frontier_T->sols.size(); s2 += NOBJS) {
						dominated = true;
						for (int p = 0; p < NOBJS && dominated; ++p) {
							dominated = ( node2->pareto_frontier_T->sols[s2+p] >= node1->pareto_frontier_T->sols[s1+p] );
						}
						if (dominated) {
							node1->pareto_frontier_T->sols[s1] = DOMINATED;
							num_dominated++;
							break;
						}
					}
					if (dominated)
						break;
				}
			}
			
			if (num_dominated > 0) {
				//cout << "\tBefore: " << node1->pareto_frontier->get_num_sols() << endl;
				node1->pareto_frontier_T->remove_dominated();
				//cout << "\tAfter: " << node1->pareto_frontier->get_num_sols() << endl << endl;				
				total += num_dominated;
			}


			//            cout << "layer: " << layer << ", node: " << node1->index << " , num_dominated: " << num_dominated << endl;
		}
	}
	int removed_nodes = 0;
	int id = 0;
	while (id < bdd->layers[layer].size()) {
		Node* node = bdd->layers[layer][id];
		if (node->pareto_frontier_S->sols.empty() && node->pareto_frontier_T->sols.empty()) {
			bdd->remove_node(node);
			bdd->layers[layer][id] = bdd->layers[layer].back();
			bdd->layers[layer].pop_back();

			++removed_nodes;
		} else {
			++id;
		}
	}

	cout << "Layer " << layer << " - total solutions filtered: " << total << " - nodes removed: " << removed_nodes << endl;
	
}


//
// Filter layer based on dominance / set packing
//
void BDDMultiObj::filter_dominance_setpacking(BDD* bdd, const int layer, MultiObjectiveStats* stats) {
    
    //	cout << "Applying filter dominance for set packing..." << endl;
    
    //     if (layer < bdd->num_layers/3 || layer > 2*bdd->num_layers/3) {
    //     	return;
    //     }
    
    //int total = 0;
    int NoNodes = (int) bdd->layers[layer].size();
    if(NoNodes > 1) {
        // Compare the nodes based on their states which are the set of variables that we can still choose
        // The supset set can potentially dominate the subset
        // First, build a dominance graph: A matrix where (i,i) = 0, and (i,j) = 1 means that i can potentially dominate j, i.e., StateSet(i) \supseteq StateSet(j)
        
        // We would like to try just one potential j to dominated states of i
        // Instead of constructing a full dominance graph, we will sort the nodes in terms of their states' size, then the smallest element and lastly the largest element, so that we can find an index j dominating i to try much faster
        // Let's call the partition based on size as "buckets"
        // Then, if i belongs to bucket k, we only need to look for j in buckets k+1,k+2,...
        // Moreover, for the elements to compare, we first check whether their smallest elements are the same. If so, then, as a last resort we call the function is_subset_of.
        // There are at most number of variables many buckets for Set Packing
        
        
        vector< vector<Node*> > Bucket_NodeIndices(bdd->num_layers-1,vector<Node*>());
        // Put the nodes into buckets
        for(int i=0; i < NoNodes; i++)
            Bucket_NodeIndices[bdd->layers[layer][i]->setpack_state.count()].push_back(bdd->layers[layer][i]);
        
        // Go over the buckets
        for(int b1=0; b1 < Bucket_NodeIndices.size()-1; b1++) {
            if(Bucket_NodeIndices[b1].size() > 1) {
                for(int i=0; i < Bucket_NodeIndices[b1].size(); i++) {
                    Node* node1 = Bucket_NodeIndices[b1][i]; // try to dominate node1
                    int num_dominated = 0;
                    bool Candidate_j_Found = false;
                    for(int b2=b1+1; b2 < Bucket_NodeIndices.size(); b2++) { // only j in a subsequent bucket can potentially dominate node1
                        for(int j=0; j < Bucket_NodeIndices[b2].size(); j++) {
                            
//                            Node* node2 = Bucket_NodeIndices[b2][j];
//                            Candidate_j_Found = true;
                            
                            Node* node2 = Bucket_NodeIndices[b2][j];    // node2 can potentially dominate node1
                            
                            // if node1 and node2 do not have smae min and max element, then they cannot be subsets
                            if(node1->setpack_state.find_first() != node2->setpack_state.find_first())
                                continue;
                            
                            // if node1 and node 2 have one parent and they are the same, continue
                            if (node1->prev[0].size() + node1->prev[1].size() == 1
                                && node2->prev[0].size() + node2->prev[1].size() == 1)
                            {
                                if (node1->prev[0].size() > 0 && node2->prev[1].size() > 0 && node1->prev[0][0] == node2->prev[1][0])
                                    continue;
                                if (node1->prev[1].size() > 0 && node2->prev[0].size() > 0 && node1->prev[1][0] == node2->prev[0][0])
                                    continue;
                            }
                            
                            if(! node1->setpack_state.is_subset_of(node2->setpack_state))
                                continue;
                            
                            Candidate_j_Found = true;
                            
                            // Check each label of node at index1 whether it is dominated or not
                            for (int s1 = 0; s1 < node1->pareto_frontier->sols.size(); s1 += NOBJS) {
                                if(node1->pareto_frontier->sols[s1] == DOMINATED)
                                    continue;
                                
                                bool dominated = true;
                                for (int s2 = 0; s2 < node2->pareto_frontier->sols.size(); s2 += NOBJS) {
                                    dominated = true;
                                    for (int p = 0; p < NOBJS && dominated; ++p)
                                        dominated = ( node2->pareto_frontier->sols[s2+p] >= node1->pareto_frontier->sols[s1+p] );
                                    if (dominated) {
                                        node1->pareto_frontier->sols[s1] = DOMINATED;
                                        num_dominated++;
                                        break;
                                    }
                                }
                            }
                            
                           if(Candidate_j_Found)
                               break;
                        }
                        
                        // TRY ONLY THE NEXT NONEMPTY BUCKET
                       if(Bucket_NodeIndices[b2].size() > 0)
                           break;
                        
                       if(Candidate_j_Found)
                           break;
                    }
                    if (num_dominated > 0) {
                        //cout << "\tBefore: " << node1->pareto_frontier->get_num_sols() << endl;
                        node1->pareto_frontier->remove_dominated();
                        //cout << "\tAfter: " << node1->pareto_frontier->get_num_sols() << endl << endl;
						//total += num_dominated;
						stats->pareto_dominance_filtered += num_dominated;
                    }
                    //            cout << "layer: " << layer << ", node: " << node1->index << " , num_dominated: " << num_dominated << endl;
                }
            }
        }
    }
    //cout << "Layer " << layer << " - total filtered: " << total << endl;
}

//void BDDMultiObj::filter_dominance_setpacking(BDD* bdd, const int layer) { // OLD VERSION where we build the ominance graph and check all pairs
//    //	cout << "Applying filter dominance for set packing..." << endl;
//    
////     if (layer < bdd->num_layers/3 || layer > 2*bdd->num_layers/3) {
////     	return;
////     }
//    
//    int total = 0;
//    int NoNodes = (int) bdd->layers[layer].size();
//    if(NoNodes > 1) {
//        // Compare the nodes based on their states which are the set of variables that we can still choose
//        // The supset set can potentially dominate the subset
//        // First, build a dominance graph: A matrix where (i,i) = 0, and (i,j) = 1 means that i can potentially dominate j, i.e., StateSet(i) \supseteq StateSet(j)
//        
//        vector< vector<int> > DominanceGraph(NoNodes,vector<int>(NoNodes,0));
//        for(int i=0; i < NoNodes-1; i++) {
//            for(int j=i+1; j < NoNodes; j++) {
//                if(bdd->layers[layer][j]->setpack_state.is_subset_of(bdd->layers[layer][i]->setpack_state))  // i is a supset of j
//                    DominanceGraph[i][j] = 1;
//                if(bdd->layers[layer][i]->setpack_state.is_subset_of(bdd->layers[layer][j]->setpack_state))  // j is a supset of i
//                    DominanceGraph[j][i] = 1;
//            }
//        }
//        
//        // Heuristically try some pairs for which the dominance graph entry is 1
//        for(int i=0; i < NoNodes; i++) { // try to dominate i
//            int index1 = i;
//            Node* node1 = bdd->layers[layer][index1];
//            int num_dominated = 0;
//            
//            for(int j=0; j < NoNodes; j++) {
//                if(DominanceGraph[j][i] == 1) {
//                    // j can potentially dominate i
//                    int index2 = j;
//                    Node* node2 = bdd->layers[layer][index2];
//                    
//                    // if node1 and node 2 have one parent and they are the same, continue
//                    if (node1->prev[0].size() + node1->prev[1].size() == 1
//                        && node2->prev[0].size() + node2->prev[1].size() == 1)
//                    {
//                        if (node1->prev[0].size() > 0 && node2->prev[1].size() > 0 && node1->prev[0][0] == node2->prev[1][0])
//                            continue;
//                        if (node1->prev[1].size() > 0 && node2->prev[0].size() > 0 && node1->prev[1][0] == node2->prev[0][0])
//                            continue;
//                    }
//                    
//                    // Check each label of node at index1 whether it is dominated or not
//                    for (int s1 = 0; s1 < node1->pareto_frontier->sols.size(); s1 += NOBJS) {
//                        if(node1->pareto_frontier->sols[s1] == DOMINATED)
//                            continue;
//                        
//                        bool dominated = true;
//                        for (int s2 = 0; s2 < node2->pareto_frontier->sols.size(); s2 += NOBJS) {
//                            dominated = true;
//                            for (int p = 0; p < NOBJS && dominated; ++p)
//                                dominated = ( node2->pareto_frontier->sols[s2+p] >= node1->pareto_frontier->sols[s1+p] );
//                            if (dominated) {
//                                node1->pareto_frontier->sols[s1] = DOMINATED;
//                                num_dominated++;
//                                break;
//                            }
//                        }
//                    }
//                }
//            }
//            if (num_dominated > 0) {
//                //cout << "\tBefore: " << node1->pareto_frontier->get_num_sols() << endl;
//                node1->pareto_frontier->remove_dominated();
//                //cout << "\tAfter: " << node1->pareto_frontier->get_num_sols() << endl << endl;
//                total += num_dominated;
//            }
//            //            cout << "layer: " << layer << ", node: " << node1->index << " , num_dominated: " << num_dominated << endl;
//        }
//    }
//    
//    cout << "Layer " << layer << " - total filtered: " << total << endl;
//}


//
// Filter layer based on dominance / set covering
//
void BDDMultiObj::filter_dominance_setcovering(BDD* bdd, const int layer, MultiObjectiveStats* stats) {
    //	cout << "Applying filter dominance for set covering..." << endl;
    
    // if (layer > bdd->num_layers/3+10) {
    // 	return;
    // }
    
    
    // It is a MINIMIZATION problem!!!
    
    //int total = 0;
    int NoNodes = (int) bdd->layers[layer].size();
    if(NoNodes > 1) {
        // Compare the nodes based on their states which are the set of constraints that we have to cover
        // The subset set can potentially dominate the subset
        // First, build a dominance graph: A matrix where (i,i) = 0, and (i,j) = 1 means that i can potentially dominate j, i.e., StateSet(i) \subseteq StateSet(j)
        
        vector< vector<int> > DominanceGraph(NoNodes,vector<int>(NoNodes,0));
        for(int i=0; i < NoNodes-1; i++) {
            for(int j=i+1; j < NoNodes; j++) {
                if(bdd->layers[layer][i]->setcover_state.is_subset_of(bdd->layers[layer][j]->setcover_state))  // i is a supset of j
                    DominanceGraph[i][j] = 1;
                if(bdd->layers[layer][j]->setcover_state.is_subset_of(bdd->layers[layer][i]->setcover_state))  // j is a supset of i
                    DominanceGraph[j][i] = 1;
            }
        }
        
        // Heuristically try some pairs for which the dominance graph entry is 1
        for(int i=0; i < NoNodes; i++) { // try to dominate i
            int index1 = i;
            Node* node1 = bdd->layers[layer][index1];
            int num_dominated = 0;
            
            for(int j=0; j < NoNodes; j++) {
                if(DominanceGraph[j][i] == 1) {
                    // j can potentially dominate i
                    int index2 = j;
                    Node* node2 = bdd->layers[layer][index2];
                    
                    // if node1 and node 2 have one parent and they are the same, continue
                    if (node1->prev[0].size() + node1->prev[1].size() == 1
                        && node2->prev[0].size() + node2->prev[1].size() == 1)
                    {
                        if (node1->prev[0].size() > 0 && node2->prev[1].size() > 0 && node1->prev[0][0] == node2->prev[1][0])
                            continue;
                        if (node1->prev[1].size() > 0 && node2->prev[0].size() > 0 && node1->prev[1][0] == node2->prev[0][0])
                            continue;
                    }
                    
                    // Check each label of node at index1 whether it is dominated or not
                    for (int s1 = 0; s1 < node1->pareto_frontier->sols.size(); s1 += NOBJS) {
                        if(node1->pareto_frontier->sols[s1] == DOMINATED)
                            continue;
                        
                        bool dominated = true;
                        for (int s2 = 0; s2 < node2->pareto_frontier->sols.size(); s2 += NOBJS) {
                            dominated = true;
                            for (int p = 0; p < NOBJS && dominated; ++p)
                                dominated = ( node2->pareto_frontier->sols[s2+p] >= node1->pareto_frontier->sols[s1+p] );
                            if (dominated) {
                                node1->pareto_frontier->sols[s1] = DOMINATED;
                                num_dominated++;
                                break;
                            }
                        }
                    }
                }
            }
            if (num_dominated > 0) {
                //cout << "\tBefore: " << node1->pareto_frontier->get_num_sols() << endl;
                node1->pareto_frontier->remove_dominated();
                //cout << "\tAfter: " << node1->pareto_frontier->get_num_sols() << endl << endl;
                stats->pareto_dominance_filtered += num_dominated;
            }
            //            cout << "layer: " << layer << ", node: " << node1->index << " , num_dominated: " << num_dominated << endl;
        }
    }
    
    //cout << "Layer " << layer << " - total filtered: " << total << endl;
}


//
// Find pareto frontier using top-down approach for MDDs
//
ParetoFrontier* BDDMultiObj::pareto_frontier_topdown(MDD* mdd, MultiObjectiveStats* stats, int cpu_threads) {
	// Initialize stats
	stats->pareto_dominance_time = 0;
	stats->pareto_dominance_filtered = 0;
    reset_cpu_perf_stats(stats);
    const bool perf_enabled = cpu_perf_enabled(stats);
    const int threads = normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = use_parallel_cpu(threads);
	
	// Initialize manager
	ParetoFrontierManager* mgmr = new ParetoFrontierManager(mdd->get_width());

	// Root node
    ObjType zero_array[NOBJS];
	memset(zero_array, 0, sizeof(ObjType)*NOBJS);
	mdd->get_root()->pareto_frontier = request_frontier(mgmr, parallel_mode);
	mdd->get_root()->pareto_frontier->add(zero_array);
    
	// Generate frontiers for each node
	for (int l = 1; l < mdd->num_layers; ++l) {	
		cout << "Layer " << l << endl;
        const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        const int layer_size = mdd->layers[l].size();
#ifdef _OPENMP
#pragma omp parallel for if(parallel_mode) num_threads(threads) schedule(dynamic)
#endif
		for (int i = 0; i < layer_size; ++i) {
            MDDNode* node = mdd->layers[l][i];
			node->pareto_frontier = request_frontier(mgmr, parallel_mode);
			for (MDDArc* arc : node->in_arcs_list) {
				node->pareto_frontier->merge(*(arc->tail->pareto_frontier), arc->weights);
			}
		}

		// Deallocate frontier from previous layer
		for (MDDNode* node : mdd->layers[l-1]) {
			recycle_frontier(mgmr, node->pareto_frontier, parallel_mode);
		}
        if (perf_enabled) {
            stats->cpu_expand_td_wall_s += wall_elapsed_s(expand_begin);
            stats->cpu_layers_td += 1;
            stats->cpu_nodes_expanded += layer_size;
        }
	}		

    ParetoFrontier* frontier = mdd->get_terminal()->pareto_frontier;

    // Erase memory
	delete mgmr;
	return frontier;
}

//
// Find pareto frontier using dynamic layer cutset on CUDA (MDD)
// kernel_version: 1=one-block-per-node, 2=fixed-2D-grid, 3=dynamic-1D-grid
//
ParetoFrontier* BDDMultiObj::pareto_frontier_dynamic_layer_cutset_cuda(MDD* mdd, MultiObjectiveStats* stats, std::string* reason, int kernel_version) {
    if (stats != NULL) {
        stats->pareto_dominance_time = 0;
        stats->pareto_dominance_filtered = 0;
        stats->layer_coupling = 0;
    }

#ifdef USE_CUDA
    if (coupled_cuda_enumerate == NULL) {
        if (reason != NULL) {
            *reason = "CUDA coupled enumeration symbol is unavailable in this binary";
        }
        return NULL;
    }

    std::string local_reason;
    std::string* active_reason = reason != NULL ? reason : &local_reason;
    ParetoFrontier* frontier = coupled_cuda_enumerate(mdd, stats, active_reason, kernel_version);
    if (frontier == NULL && reason != NULL && reason->empty()) {
        *reason = "CUDA coupled enumeration failed";
    }
    return frontier;
#else
    (void)mdd;
    (void)kernel_version;
    if (reason != NULL) {
        *reason = "GPU backend requested but binary was built without CUDA support";
    }
    return NULL;
#endif
}


//
// Find pareto frontier using dynamic layer cutset
//
ParetoFrontier* BDDMultiObj::pareto_frontier_dynamic_layer_cutset(MDD* mdd, MultiObjectiveStats* stats, int cpu_threads) {
	// Create pareto frontier manager
	ParetoFrontierManager* mgmr = new ParetoFrontierManager(mdd->get_width());
    const int threads = normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = use_parallel_cpu(threads);
    reset_cpu_perf_stats(stats);
    const bool perf_enabled = cpu_perf_enabled(stats);

	// Create root and terminal frontiers
	ObjType sol[NOBJS];
	memset(sol, 0, sizeof(ObjType)*NOBJS);

	mdd->get_root()->pareto_frontier = request_frontier(mgmr, parallel_mode);
	mdd->get_root()->pareto_frontier->add(sol);

	mdd->get_terminal()->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);
	mdd->get_terminal()->pareto_frontier_bu->add(sol);

	// Initialize stats
	stats->pareto_dominance_time = 0;
	stats->pareto_dominance_filtered = 0;

	// Current layers
	int layer_topdown = 0;
	int layer_bottomup = mdd->num_layers-1;

	// Value of layer
	int val_topdown = 0;
	int val_bottomup = 0;

	while (layer_topdown != layer_bottomup) {
		// cout << "Layer topdown: " << layer_topdown << " - layer bottomup: " << layer_bottomup << endl;
		if (val_topdown <= val_bottomup) {
            // Expand topdown
            const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
			expand_layer_topdown(mdd, ++layer_topdown, mgmr, threads);
            if (perf_enabled) {
                stats->cpu_expand_td_wall_s += wall_elapsed_s(expand_begin);
                stats->cpu_layers_td += 1;
                stats->cpu_nodes_expanded += mdd->layers[layer_topdown].size();
            }
			// Recompute layer value
			val_topdown = 0;
            const int layer_size = mdd->layers[layer_topdown].size();
            const WallClock::time_point recompute_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
#ifdef _OPENMP
#pragma omp parallel for if(parallel_mode) num_threads(threads) reduction(+:val_topdown)
#endif
			for (int i = 0; i < layer_size; ++i) {
				val_topdown += topdown_layer_value(mdd, mdd->layers[layer_topdown][i]);
			}
            if (perf_enabled) {
                stats->cpu_recompute_td_wall_s += wall_elapsed_s(recompute_begin);
            }
		} else {
			// Expand layer bottomup
            const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
			expand_layer_bottomup(mdd, --layer_bottomup, mgmr, threads);
            if (perf_enabled) {
                stats->cpu_expand_bu_wall_s += wall_elapsed_s(expand_begin);
                stats->cpu_layers_bu += 1;
                stats->cpu_nodes_expanded += mdd->layers[layer_bottomup].size();
            }
			// Recompute layer value
			val_bottomup = 0;
            const int layer_size = mdd->layers[layer_bottomup].size();
            const WallClock::time_point recompute_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
#ifdef _OPENMP
#pragma omp parallel for if(parallel_mode) num_threads(threads) reduction(+:val_bottomup)
#endif
			for (int i = 0; i < layer_size; ++i) {
				val_bottomup += bottomup_layer_value(mdd, mdd->layers[layer_bottomup][i]);
			}
            if (perf_enabled) {
                stats->cpu_recompute_bu_wall_s += wall_elapsed_s(recompute_begin);
            }
		}
	}

	// Save stats
	stats->layer_coupling = layer_topdown;

	vector<MDDNode*>& cutset = mdd->layers[layer_topdown];	
    if (perf_enabled) {
        stats->cpu_cutset_size = cutset.size();
    }
    const WallClock::time_point cutset_sort_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
	sort(cutset.begin(), cutset.end(), CompareMDDNode());
    if (perf_enabled) {
        stats->cpu_cutset_sort_wall_s += wall_elapsed_s(cutset_sort_begin);
    }

	// Compute expected frontier size
	long int expected_size = 0;
	for (int i = 0; i < cutset.size(); ++i) {
		expected_size += cutset[i]->pareto_frontier->get_num_sols() 
				* cutset[i]->pareto_frontier_bu->get_num_sols(); 
	}
	expected_size = 10000;

	ParetoFrontier* paretoFrontier = new ParetoFrontier;
	paretoFrontier->sols.reserve( expected_size * NOBJS );

    if (parallel_mode && cutset.size() > 1) {
#ifdef _OPENMP
        const WallClock::time_point convolution_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        vector<ParetoFrontier*> partial(threads, NULL);
#pragma omp parallel num_threads(threads)
        {
            const int tid = omp_get_thread_num();
            ParetoFrontier* local_frontier = new ParetoFrontier;
            partial[tid] = local_frontier;
#pragma omp for schedule(dynamic)
            for (int i = 0; i < cutset.size(); ++i) {
                MDDNode* node = cutset[i];
                assert( node->pareto_frontier != NULL );
                assert( node->pareto_frontier_bu != NULL );
                local_frontier->convolute( *(node->pareto_frontier), *(node->pareto_frontier_bu) );
            }
        }
        if (perf_enabled) {
            stats->cpu_cutset_convolution_wall_s += wall_elapsed_s(convolution_begin);
        }
        const WallClock::time_point partial_merge_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        for (int t = 0; t < threads; ++t) {
            if (partial[t] != NULL) {
                paretoFrontier->merge(*partial[t]);
                delete partial[t];
            }
        }
        if (perf_enabled) {
            stats->cpu_cutset_partial_merge_wall_s += wall_elapsed_s(partial_merge_begin);
        }
#else
        const WallClock::time_point convolution_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        for (int i = 0; i < cutset.size(); ++i) {
            MDDNode* node = cutset[i];
            assert( node->pareto_frontier != NULL );
            assert( node->pareto_frontier_bu != NULL );
            paretoFrontier->convolute( *(node->pareto_frontier), *(node->pareto_frontier_bu) );
        }
        if (perf_enabled) {
            stats->cpu_cutset_convolution_wall_s += wall_elapsed_s(convolution_begin);
        }
#endif
    } else {
        const WallClock::time_point convolution_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        for (int i = 0; i < cutset.size(); ++i) {
            MDDNode* node = cutset[i];
            assert( node->pareto_frontier != NULL );
            assert( node->pareto_frontier_bu != NULL );
            paretoFrontier->convolute( *(node->pareto_frontier), *(node->pareto_frontier_bu) );
        }
        if (perf_enabled) {
            stats->cpu_cutset_convolution_wall_s += wall_elapsed_s(convolution_begin);
        }
    }
    
    // deallocate manager
	delete mgmr;

    // return pareto frontier
	return paretoFrontier;
}
