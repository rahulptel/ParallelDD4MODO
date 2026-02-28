// ----------------------------------------------------------
// BDD Multiobjective Algorithms - Implementation
// ----------------------------------------------------------

#include "bdd_multiobj.hpp"
#include "bdd_alg.hpp"
#include <chrono>

#include "../cuda/cuda_wrappers.hpp"
#include "../util/omp_compat.hpp"

typedef std::pair<int,int> intpair;

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

inline bool cpu_perf_enabled(const EnumerationStats* stats) {
    return (stats != NULL && stats->cpu_perf_enabled);
}

inline double wall_elapsed_s(const WallClock::time_point& start) {
    return std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - start).count();
}

inline void reset_cpu_perf_stats(EnumerationStats* stats) {
    if (stats == NULL) {
        return;
    }
    stats->wall_expand_td_s = 0.0;
    stats->wall_expand_bu_s = 0.0;
    stats->wall_recompute_td_s = 0.0;
    stats->wall_recompute_bu_s = 0.0;
    stats->wall_dominance_s = 0.0;
    stats->wall_cutset_sort_s = 0.0;
    stats->wall_cutset_convolution_s = 0.0;
    stats->wall_cutset_partial_merge_s = 0.0;
    stats->wall_pack_transfer_s = 0.0;
    stats->wall_join_s = 0.0;
    stats->kernel_expand_td_s = 0.0;
    stats->kernel_dominance_s = 0.0;
    stats->kernel_total_s = 0.0;
    stats->gpu_mem_peak_used_bytes = 0;
    stats->gpu_mem_peak_reserved_bytes = 0;
    stats->work_candidates_total = 0;
    stats->work_frontier_survivors_total = 0;
    stats->work_frontier_peak_points = 0;
    stats->work_join_products_total = 0;
    stats->cpu_layers_td = 0;
    stats->cpu_layers_bu = 0;
    stats->cpu_nodes_expanded = 0;
    stats->cpu_cutset_size = 0;
}

inline void update_peak_points(EnumerationStats* stats, const long long value) {
    if (stats == NULL) {
        return;
    }
    if (value > stats->work_frontier_peak_points) {
        stats->work_frontier_peak_points = value;
    }
}

inline long long count_bdd_candidates_topdown_layer(BDD* bdd, const int layer, const bool maximization) {
    long long total = 0;
    const int first_arc_type = maximization ? 1 : 0;
    const int second_arc_type = maximization ? 0 : 1;
    for (Node* node : bdd->layers[layer]) {
        for (Node* prev : node->prev[first_arc_type]) {
            if (prev != NULL && prev->pareto_frontier != NULL) {
                total += prev->pareto_frontier->get_num_sols();
            }
        }
        for (Node* prev : node->prev[second_arc_type]) {
            if (prev != NULL && prev->pareto_frontier != NULL) {
                total += prev->pareto_frontier->get_num_sols();
            }
        }
    }
    return total;
}

inline long long count_bdd_survivors_topdown_layer(BDD* bdd, const int layer) {
    long long total = 0;
    for (Node* node : bdd->layers[layer]) {
        if (node != NULL && node->pareto_frontier != NULL) {
            total += node->pareto_frontier->get_num_sols();
        }
    }
    return total;
}

inline long long count_bdd_candidates_bottomup_layer(BDD* bdd, const int layer, const bool maximization) {
    long long total = 0;
    const int first_arc_type = maximization ? 1 : 0;
    const int second_arc_type = maximization ? 0 : 1;
    for (Node* node : bdd->layers[layer]) {
        if (node->arcs[first_arc_type] != NULL && node->arcs[first_arc_type]->pareto_frontier_bu != NULL) {
            total += node->arcs[first_arc_type]->pareto_frontier_bu->get_num_sols();
        }
        if (node->arcs[second_arc_type] != NULL && node->arcs[second_arc_type]->pareto_frontier_bu != NULL) {
            total += node->arcs[second_arc_type]->pareto_frontier_bu->get_num_sols();
        }
    }
    return total;
}

inline long long count_bdd_survivors_bottomup_layer(BDD* bdd, const int layer) {
    long long total = 0;
    for (Node* node : bdd->layers[layer]) {
        if (node != NULL && node->pareto_frontier_bu != NULL) {
            total += node->pareto_frontier_bu->get_num_sols();
        }
    }
    return total;
}

inline long long count_mdd_candidates_topdown_layer(MDD* mdd, const int layer) {
    long long total = 0;
    for (MDDNode* node : mdd->layers[layer]) {
        for (MDDArc* arc : node->in_arcs_list) {
            if (arc != NULL && arc->tail != NULL && arc->tail->pareto_frontier != NULL) {
                total += arc->tail->pareto_frontier->get_num_sols();
            }
        }
    }
    return total;
}

inline long long count_mdd_survivors_topdown_layer(MDD* mdd, const int layer) {
    long long total = 0;
    for (MDDNode* node : mdd->layers[layer]) {
        if (node != NULL && node->pareto_frontier != NULL) {
            total += node->pareto_frontier->get_num_sols();
        }
    }
    return total;
}

inline long long count_mdd_candidates_bottomup_layer(MDD* mdd, const int layer) {
    long long total = 0;
    for (MDDNode* node : mdd->layers[layer]) {
        for (MDDArc* arc : node->out_arcs_list) {
            if (arc != NULL && arc->head != NULL && arc->head->pareto_frontier_bu != NULL) {
                total += arc->head->pareto_frontier_bu->get_num_sols();
            }
        }
    }
    return total;
}

inline long long count_mdd_survivors_bottomup_layer(MDD* mdd, const int layer) {
    long long total = 0;
    for (MDDNode* node : mdd->layers[layer]) {
        if (node != NULL && node->pareto_frontier_bu != NULL) {
            total += node->pareto_frontier_bu->get_num_sols();
        }
    }
    return total;
}

//
// Find pareto frontier using top-down approach on CUDA
// kernel_version: 1=one-block-per-node, 2=fixed-2D-grid, 3=dynamic-1D-grid
//
ParetoFrontier* BDDMultiObj::pareto_frontier_topdown_cuda(BDD* bdd, bool maximization, const int problem_type, const int dominance_strategy, EnumerationStats* stats, std::string* reason, int kernel_version) {
    if (stats != NULL) {
        stats->cpu_ticks_dominance = 0;
        stats->dominance_filtered_total = 0;
        stats->layer_coupling = 0;
        reset_cpu_perf_stats(stats);
    }

    ParetoFrontier* frontier = ::topdown_cuda_enumerate(bdd, maximization, problem_type, dominance_strategy, stats, reason, kernel_version);
    return frontier;
}

//
// Find pareto frontier using top-down approach on CUDA for MDD
// kernel_version: 1=one-block-per-node, 2=fixed-2D-grid, 3=dynamic-1D-grid
//
ParetoFrontier* BDDMultiObj::pareto_frontier_topdown_cuda(MDD* mdd, EnumerationStats* stats, std::string* reason, int kernel_version) {
    if (stats != NULL) {
        stats->cpu_ticks_dominance = 0;
        stats->dominance_filtered_total = 0;
        stats->layer_coupling = 0;
        reset_cpu_perf_stats(stats);
    }

    ParetoFrontier* frontier = ::topdown_mdd_cuda_enumerate(mdd, stats, reason, kernel_version);
    return frontier;
}

//
// Find pareto frontier using top-down approach
//
ParetoFrontier* BDDMultiObj::pareto_frontier_topdown(BDD* bdd, bool maximization, const int problem_type, int dominance_strategy, EnumerationStats* stats, int cpu_threads) {
    //cout << "\nComputing Pareto Frontier..." << endl;

	// Initialize stats
	stats->cpu_ticks_dominance = 0;
	stats->dominance_filtered_total = 0;
    reset_cpu_perf_stats(stats);
    clock_t init;
    const bool perf_enabled = cpu_perf_enabled(stats);
	
	// Initialize manager
	ParetoFrontierManager* mgmr = new ParetoFrontierManager(bdd->get_width());
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);

	// root node
    ObjType zero_array[NOBJS];
	memset(zero_array, 0, sizeof(ObjType)*NOBJS);
	bdd->get_root()->pareto_frontier = request_frontier(mgmr, parallel_mode);
	bdd->get_root()->pareto_frontier->add(zero_array);

    
	if (maximization) {
		for (int l = 1; l < bdd->num_layers; ++l) {
//			cout << "\tLayer " << l << " - size = " << bdd->layers[l].size() << '\n';
            const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
            const long long layer_candidates = count_bdd_candidates_topdown_layer(bdd, l, maximization);
		
            const int layer_size = bdd->layers[l].size();
            CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
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
                    stats->cpu_ticks_dominance += clock()-init;
                    if (perf_enabled) {
                        stats->wall_dominance_s += wall_elapsed_s(dominance_begin);
                    }
				}

                const long long layer_survivors = count_bdd_survivors_topdown_layer(bdd, l);
                if (stats != NULL) {
                    stats->work_candidates_total += layer_candidates;
                    stats->work_frontier_survivors_total += layer_survivors;
                    update_peak_points(stats, layer_survivors);
                }
				
			// Deallocate frontier from previous layer
				for (vector<Node*>::iterator it = bdd->layers[l-1].begin(); it != bdd->layers[l-1].end(); ++it) {
					recycle_frontier(mgmr, (*it)->pareto_frontier, parallel_mode);
				}
                if (perf_enabled) {
                    stats->wall_expand_td_s += wall_elapsed_s(expand_begin);
                    stats->cpu_layers_td += 1;
                    stats->cpu_nodes_expanded += layer_size;
                }
			}
		} else {
			for (int l = 1; l < bdd->num_layers; ++l) {
				//cout << "\tLayer " << l << " - size = " << bdd->layers[l].size() << '\n';
                const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
                const long long layer_candidates = count_bdd_candidates_topdown_layer(bdd, l, maximization);
		
            const int layer_size = bdd->layers[l].size();
            CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
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
					stats->cpu_ticks_dominance += clock()-init;
                    if (perf_enabled) {
                        stats->wall_dominance_s += wall_elapsed_s(dominance_begin);
                    }
				}

                const long long layer_survivors = count_bdd_survivors_topdown_layer(bdd, l);
                if (stats != NULL) {
                    stats->work_candidates_total += layer_candidates;
                    stats->work_frontier_survivors_total += layer_survivors;
                    update_peak_points(stats, layer_survivors);
                }

			// Deallocate frontier from previous layer
				for (vector<Node*>::iterator it = bdd->layers[l-1].begin(); it != bdd->layers[l-1].end(); ++it) {
					recycle_frontier(mgmr, (*it)->pareto_frontier, parallel_mode);
				}
                if (perf_enabled) {
                    stats->wall_expand_td_s += wall_elapsed_s(expand_begin);
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
ParetoFrontier* BDDMultiObj::pareto_frontier_bottomup(BDD* bdd, bool maximization, const int problem_type, const int dominance_strategy, EnumerationStats* stats, int cpu_threads) {
    //cout << "\nComputing Pareto Set...\n";
	(void)problem_type;
	(void)dominance_strategy;
	reset_cpu_perf_stats(stats);
    const bool perf_enabled = cpu_perf_enabled(stats);
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);

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
            const long long layer_candidates = count_bdd_candidates_bottomup_layer(bdd, l, maximization);

            const int layer_size = bdd->layers[l].size();
            CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
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
                const long long layer_survivors = count_bdd_survivors_bottomup_layer(bdd, l);
                if (stats != NULL) {
                    stats->work_candidates_total += layer_candidates;
                    stats->work_frontier_survivors_total += layer_survivors;
                    update_peak_points(stats, layer_survivors);
                }
                if (perf_enabled) {
                    stats->wall_expand_bu_s += wall_elapsed_s(expand_begin);
                    stats->cpu_layers_bu += 1;
                    stats->cpu_nodes_expanded += layer_size;
                }
			} 
		} else {
			for (int l = bdd->num_layers-2; l >= 0; --l) {
				// cout << "\tLayer " << l << " - size = " << bdd->layers[l].size();
				// cout << '\n';
                const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
                const long long layer_candidates = count_bdd_candidates_bottomup_layer(bdd, l, maximization);

            const int layer_size = bdd->layers[l].size();
            CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
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
                const long long layer_survivors = count_bdd_survivors_bottomup_layer(bdd, l);
                if (stats != NULL) {
                    stats->work_candidates_total += layer_candidates;
                    stats->work_frontier_survivors_total += layer_survivors;
                    update_peak_points(stats, layer_survivors);
                }
                if (perf_enabled) {
                    stats->wall_expand_bu_s += wall_elapsed_s(expand_begin);
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
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);
	if (maximization) {
        const int layer_size = bdd->layers[l].size();
            CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
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
            CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
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
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);
    const int layer_size = mdd->layers[l].size();
            CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
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
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);
    const int layer_size = mdd->layers[l].size();
            CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
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
ParetoFrontier* BDDMultiObj::pareto_frontier_dynamic_layer_cutset(BDD* bdd, bool maximization, const int problem_type, const int dominance_strategy, EnumerationStats* stats, int cpu_threads) {
	// Create pareto frontier manager
	ParetoFrontierManager* mgmr = new ParetoFrontierManager(bdd->get_width());
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);
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
	stats->cpu_ticks_dominance = 0;
	stats->dominance_filtered_total = 0;
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
            const int next_layer = layer_topdown + 1;
            const long long layer_candidates = count_bdd_candidates_topdown_layer(bdd, next_layer, maximization);
            // Expand topdown
            const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
			expand_layer_topdown(bdd, ++layer_topdown, maximization, mgmr, threads);
            if (perf_enabled) {
                stats->wall_expand_td_s += wall_elapsed_s(expand_begin);
                stats->cpu_layers_td += 1;
                stats->cpu_nodes_expanded += bdd->layers[layer_topdown].size();
            }
			// Recompute layer value
			val_topdown = 0;
            const int layer_size = bdd->layers[layer_topdown].size();
            const WallClock::time_point recompute_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
            CUMODD_OMP_PARALLEL_FOR_REDUCTION_SUM_IF(parallel_mode, threads, val_topdown)
			for (int i = 0; i < layer_size; ++i) {
				val_topdown += topdown_layer_value(bdd, bdd->layers[layer_topdown][i]);
			}
            if (perf_enabled) {
                stats->wall_recompute_td_s += wall_elapsed_s(recompute_begin);
            }
			//cout << "DOMINANCE: " << dominance_strategy << endl;
            if (dominance_strategy > 0) {
                const WallClock::time_point dominance_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
                init = clock();
				BDDMultiObj::filter_dominance(bdd, layer_topdown, problem_type, dominance_strategy, stats);
                stats->cpu_ticks_dominance += clock()-init;
                if (perf_enabled) {
                    stats->wall_dominance_s += wall_elapsed_s(dominance_begin);
                }
			}
            const long long layer_survivors = count_bdd_survivors_topdown_layer(bdd, layer_topdown);
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                stats->work_frontier_survivors_total += layer_survivors;
                update_peak_points(stats, layer_survivors);
            }
		} else {
            const int next_layer = layer_bottomup - 1;
            const long long layer_candidates = count_bdd_candidates_bottomup_layer(bdd, next_layer, maximization);
			// Expand layer bottomup
            const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
			expand_layer_bottomup(bdd, --layer_bottomup, maximization, mgmr, threads);
            if (perf_enabled) {
                stats->wall_expand_bu_s += wall_elapsed_s(expand_begin);
                stats->cpu_layers_bu += 1;
                stats->cpu_nodes_expanded += bdd->layers[layer_bottomup].size();
            }
			// Recompute layer value
			val_bottomup = 0;
            const int layer_size = bdd->layers[layer_bottomup].size();
            const WallClock::time_point recompute_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
            CUMODD_OMP_PARALLEL_FOR_REDUCTION_SUM_IF(parallel_mode, threads, val_bottomup)
			for (int i = 0; i < layer_size; ++i) {
				val_bottomup += bottomup_layer_value(bdd, bdd->layers[layer_bottomup][i]);
			}
            if (perf_enabled) {
                stats->wall_recompute_bu_s += wall_elapsed_s(recompute_begin);
            }
            const long long layer_survivors = count_bdd_survivors_bottomup_layer(bdd, layer_bottomup);
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                stats->work_frontier_survivors_total += layer_survivors;
                update_peak_points(stats, layer_survivors);
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
        stats->wall_cutset_sort_s += wall_elapsed_s(cutset_sort_begin);
    }

	// Compute expected frontier size
	long int expected_size = 0;
	for (int i = 0; i < cutset.size(); ++i) {
		expected_size += cutset[i]->pareto_frontier->get_num_sols() 
				* cutset[i]->pareto_frontier_bu->get_num_sols(); 
	}
    if (stats != NULL) {
        stats->work_join_products_total += expected_size;
    }
	expected_size = 10000;

	ParetoFrontier* paretoFrontier = new ParetoFrontier;
	paretoFrontier->sols.reserve( expected_size * NOBJS );

    if (parallel_mode && cutset.size() > 1) {
        const WallClock::time_point join_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        const WallClock::time_point convolution_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        vector<ParetoFrontier*> partial(threads, NULL);
        CUMODD_OMP_PARALLEL_NUM_THREADS(threads)
        {
            const int tid = cumodd_omp_thread_num();
            ParetoFrontier* local_frontier = new ParetoFrontier;
            partial[tid] = local_frontier;
            CUMODD_OMP_FOR_DYNAMIC
            for (int i = 0; i < cutset.size(); ++i) {
                Node* node = cutset[i];
                assert( node->pareto_frontier != NULL );
                assert( node->pareto_frontier_bu != NULL );
                local_frontier->convolute( *(node->pareto_frontier), *(node->pareto_frontier_bu) );
            }
        }
        if (perf_enabled) {
            stats->wall_cutset_convolution_s += wall_elapsed_s(convolution_begin);
        }
        const WallClock::time_point partial_merge_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        for (int t = 0; t < threads; ++t) {
            if (partial[t] != NULL) {
                paretoFrontier->merge(*partial[t]);
                delete partial[t];
            }
        }
        if (perf_enabled) {
            stats->wall_cutset_partial_merge_s += wall_elapsed_s(partial_merge_begin);
            stats->wall_join_s += wall_elapsed_s(join_begin);
        }
    } else {
        const WallClock::time_point join_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        const WallClock::time_point convolution_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        for (int i = 0; i < cutset.size(); ++i) {
            Node* node = cutset[i];
            assert( node->pareto_frontier != NULL );
            assert( node->pareto_frontier_bu != NULL );
            paretoFrontier->convolute( *(node->pareto_frontier), *(node->pareto_frontier_bu) );
        }
        if (perf_enabled) {
            stats->wall_cutset_convolution_s += wall_elapsed_s(convolution_begin);
            stats->wall_join_s += wall_elapsed_s(join_begin);
        }
    }
    if (stats != NULL) {
        const long long join_survivors = paretoFrontier->get_num_sols();
        stats->work_frontier_survivors_total += join_survivors;
        update_peak_points(stats, join_survivors);
    }
    
    // deallocate manager
	delete mgmr;

    // return pareto frontier
	return paretoFrontier;
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
inline void BDDMultiObj::filter_dominance(BDD* bdd, const int layer, const int problem_type, const int dominance_strategy, EnumerationStats* stats) {
	if (problem_type == 1) {
		// Knapsack
		filter_dominance_knapsack(bdd, layer, stats);
	} else if (problem_type == 2) {
		// Set packing
        filter_dominance_setpacking(bdd, layer, stats);
	}
}



//
// Filter layer based on dominance / knapsack
//
void BDDMultiObj::filter_dominance_knapsack(BDD* bdd, const int layer, EnumerationStats* stats) {
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
				stats->dominance_filtered_total += num_dominated;
            }
//            cout << "layer: " << layer << ", node: " << node1->index << " , num_dominated: " << num_dominated << endl;
        }
	}
	//cout << "Layer " << layer << " - total filtered: " << total << endl;
}
	

//
// Filter layer based on dominance / set packing
//
void BDDMultiObj::filter_dominance_setpacking(BDD* bdd, const int layer, EnumerationStats* stats) {
    
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
						stats->dominance_filtered_total += num_dominated;
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
// Find pareto frontier using top-down approach for MDDs
//
ParetoFrontier* BDDMultiObj::pareto_frontier_topdown(MDD* mdd, EnumerationStats* stats, int cpu_threads) {
	// Initialize stats
	stats->cpu_ticks_dominance = 0;
	stats->dominance_filtered_total = 0;
    reset_cpu_perf_stats(stats);
    const bool perf_enabled = cpu_perf_enabled(stats);
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);
	
	// Initialize manager
	ParetoFrontierManager* mgmr = new ParetoFrontierManager(mdd->get_width());

	// Root node
    ObjType zero_array[NOBJS];
	memset(zero_array, 0, sizeof(ObjType)*NOBJS);
	mdd->get_root()->pareto_frontier = request_frontier(mgmr, parallel_mode);
	mdd->get_root()->pareto_frontier->add(zero_array);
    
	// Generate frontiers for each node
	for (int l = 1; l < mdd->num_layers; ++l) {	
        const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        const long long layer_candidates = count_mdd_candidates_topdown_layer(mdd, l);
        const int layer_size = mdd->layers[l].size();
            CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
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
        const long long layer_survivors = count_mdd_survivors_topdown_layer(mdd, l);
        if (stats != NULL) {
            stats->work_candidates_total += layer_candidates;
            stats->work_frontier_survivors_total += layer_survivors;
            update_peak_points(stats, layer_survivors);
        }
        if (perf_enabled) {
            stats->wall_expand_td_s += wall_elapsed_s(expand_begin);
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
ParetoFrontier* BDDMultiObj::pareto_frontier_dynamic_layer_cutset_cuda(MDD* mdd, EnumerationStats* stats, std::string* reason, int kernel_version) {
    if (stats != NULL) {
        stats->cpu_ticks_dominance = 0;
        stats->dominance_filtered_total = 0;
        stats->layer_coupling = 0;
        reset_cpu_perf_stats(stats);
    }

    ParetoFrontier* frontier = ::coupled_cuda_enumerate(mdd, stats, reason, kernel_version);
    return frontier;
}


//
// Find pareto frontier using dynamic layer cutset
//
ParetoFrontier* BDDMultiObj::pareto_frontier_dynamic_layer_cutset(MDD* mdd, EnumerationStats* stats, int cpu_threads) {
	// Create pareto frontier manager
	ParetoFrontierManager* mgmr = new ParetoFrontierManager(mdd->get_width());
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);
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
	stats->cpu_ticks_dominance = 0;
	stats->dominance_filtered_total = 0;

	// Current layers
	int layer_topdown = 0;
	int layer_bottomup = mdd->num_layers-1;

	// Value of layer
	int val_topdown = 0;
	int val_bottomup = 0;

	while (layer_topdown != layer_bottomup) {
		// cout << "Layer topdown: " << layer_topdown << " - layer bottomup: " << layer_bottomup << endl;
		if (val_topdown <= val_bottomup) {
            const int next_layer = layer_topdown + 1;
            const long long layer_candidates = count_mdd_candidates_topdown_layer(mdd, next_layer);
            // Expand topdown
            const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
			expand_layer_topdown(mdd, ++layer_topdown, mgmr, threads);
            if (perf_enabled) {
                stats->wall_expand_td_s += wall_elapsed_s(expand_begin);
                stats->cpu_layers_td += 1;
                stats->cpu_nodes_expanded += mdd->layers[layer_topdown].size();
            }
			// Recompute layer value
			val_topdown = 0;
            const int layer_size = mdd->layers[layer_topdown].size();
            const WallClock::time_point recompute_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
            CUMODD_OMP_PARALLEL_FOR_REDUCTION_SUM_IF(parallel_mode, threads, val_topdown)
			for (int i = 0; i < layer_size; ++i) {
				val_topdown += topdown_layer_value(mdd, mdd->layers[layer_topdown][i]);
			}
            if (perf_enabled) {
                stats->wall_recompute_td_s += wall_elapsed_s(recompute_begin);
            }
            const long long layer_survivors = count_mdd_survivors_topdown_layer(mdd, layer_topdown);
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                stats->work_frontier_survivors_total += layer_survivors;
                update_peak_points(stats, layer_survivors);
            }
		} else {
            const int next_layer = layer_bottomup - 1;
            const long long layer_candidates = count_mdd_candidates_bottomup_layer(mdd, next_layer);
			// Expand layer bottomup
            const WallClock::time_point expand_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
			expand_layer_bottomup(mdd, --layer_bottomup, mgmr, threads);
            if (perf_enabled) {
                stats->wall_expand_bu_s += wall_elapsed_s(expand_begin);
                stats->cpu_layers_bu += 1;
                stats->cpu_nodes_expanded += mdd->layers[layer_bottomup].size();
            }
			// Recompute layer value
			val_bottomup = 0;
            const int layer_size = mdd->layers[layer_bottomup].size();
            const WallClock::time_point recompute_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
            CUMODD_OMP_PARALLEL_FOR_REDUCTION_SUM_IF(parallel_mode, threads, val_bottomup)
			for (int i = 0; i < layer_size; ++i) {
				val_bottomup += bottomup_layer_value(mdd, mdd->layers[layer_bottomup][i]);
			}
            if (perf_enabled) {
                stats->wall_recompute_bu_s += wall_elapsed_s(recompute_begin);
            }
            const long long layer_survivors = count_mdd_survivors_bottomup_layer(mdd, layer_bottomup);
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                stats->work_frontier_survivors_total += layer_survivors;
                update_peak_points(stats, layer_survivors);
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
        stats->wall_cutset_sort_s += wall_elapsed_s(cutset_sort_begin);
    }

	// Compute expected frontier size
	long int expected_size = 0;
	for (int i = 0; i < cutset.size(); ++i) {
		expected_size += cutset[i]->pareto_frontier->get_num_sols() 
				* cutset[i]->pareto_frontier_bu->get_num_sols(); 
	}
    if (stats != NULL) {
        stats->work_join_products_total += expected_size;
    }
	expected_size = 10000;

	ParetoFrontier* paretoFrontier = new ParetoFrontier;
	paretoFrontier->sols.reserve( expected_size * NOBJS );

    if (parallel_mode && cutset.size() > 1) {
        const WallClock::time_point join_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        const WallClock::time_point convolution_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        vector<ParetoFrontier*> partial(threads, NULL);
        CUMODD_OMP_PARALLEL_NUM_THREADS(threads)
        {
            const int tid = cumodd_omp_thread_num();
            ParetoFrontier* local_frontier = new ParetoFrontier;
            partial[tid] = local_frontier;
            CUMODD_OMP_FOR_DYNAMIC
            for (int i = 0; i < cutset.size(); ++i) {
                MDDNode* node = cutset[i];
                assert( node->pareto_frontier != NULL );
                assert( node->pareto_frontier_bu != NULL );
                local_frontier->convolute( *(node->pareto_frontier), *(node->pareto_frontier_bu) );
            }
        }
        if (perf_enabled) {
            stats->wall_cutset_convolution_s += wall_elapsed_s(convolution_begin);
        }
        const WallClock::time_point partial_merge_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        for (int t = 0; t < threads; ++t) {
            if (partial[t] != NULL) {
                paretoFrontier->merge(*partial[t]);
                delete partial[t];
            }
        }
        if (perf_enabled) {
            stats->wall_cutset_partial_merge_s += wall_elapsed_s(partial_merge_begin);
            stats->wall_join_s += wall_elapsed_s(join_begin);
        }
    } else {
        const WallClock::time_point join_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        const WallClock::time_point convolution_begin = perf_enabled ? WallClock::now() : WallClock::time_point();
        for (int i = 0; i < cutset.size(); ++i) {
            MDDNode* node = cutset[i];
            assert( node->pareto_frontier != NULL );
            assert( node->pareto_frontier_bu != NULL );
            paretoFrontier->convolute( *(node->pareto_frontier), *(node->pareto_frontier_bu) );
        }
        if (perf_enabled) {
            stats->wall_cutset_convolution_s += wall_elapsed_s(convolution_begin);
            stats->wall_join_s += wall_elapsed_s(join_begin);
        }
    }
    if (stats != NULL) {
        const long long join_survivors = paretoFrontier->get_num_sols();
        stats->work_frontier_survivors_total += join_survivors;
        update_peak_points(stats, join_survivors);
    }
    
    // deallocate manager
	delete mgmr;

    // return pareto frontier
	return paretoFrontier;
}
