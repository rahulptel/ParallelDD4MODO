// --------------------------------------------------
// Multiobjective
// --------------------------------------------------

// General includes
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <chrono>
#include <iomanip>
#include <fstream>

#include "bdd/bdd.hpp"
#include "bdd/bdd_alg.hpp"
#include "bdd/bdd_multiobj.hpp"
#include "util/cli_parser.hpp"
#include "util/stats.hpp"
#include "util/util.hpp"
#include "util/output_utils.hpp"
#include "bdd/pareto_frontier.hpp"

// Knapsack includes
#include "instances/knapsack_instance.hpp"
#include "bdd/knapsack_bdd.hpp"

// Set packing / Independent set includes
#include "instances/indepset_instance.hpp"
#include "instances/setpacking_instance.hpp"
#include "bdd/indepset_bdd.hpp"

// TSP instance
#include "instances/tsp_instance.hpp"
#include "mdd/tsp_mdd.hpp"

using namespace std;



//
// Main function
//
int main(int argc, char *argv[])
{
    CliOptions options;
    string parse_error;
    if (!parse_cli_args(argc, argv, &options, &parse_error))
    {
        if (!parse_error.empty())
        {
            cout << parse_error << endl;
        }
        print_usage();
        exit(1);
    }

    const string input_path = options.input_path;
    const int problem_type = options.problem_type;
    const int method = options.method;
    bool maximization = true;
    const int dominance = options.dominance;
    const Backend backend = options.backend;
    const int kernel_version = options.kernel_version;
    const int cpu_threads = options.cpu_threads;
    const bool save_frontier = options.save_frontier;
    const string frontier_out_path = options.frontier_out_path;
    const bool save_stats = options.save_stats;
    const string stats_out_path = options.stats_out_path;

    typedef std::chrono::steady_clock WallClock;
    const WallClock::time_point run_wall_begin = WallClock::now();
    double compilation_wall_s = 0.0;
    double pareto_wall_enumeration_s = 0.0;

    // For statistical analysis
    TimeStats timers;
    int bdd_compilation_time = timers.register_name("BDD compilation time");
    int pareto_time = timers.register_name("BDD pareto time");
    long int original_width;
    long int reduced_width;
    long int original_num_nodes;
    long int reduced_num_nodes;

    // Read problem instance and construct BDD
    BDD *bdd = NULL;
    vector<vector<int>> obj_coeffs;
    const WallClock::time_point compilation_wall_begin = WallClock::now();
    timers.start_timer(bdd_compilation_time);

    // --- Knapsack ---
    if (problem_type == 1)
    {

        // Read instance
        KnapsackInstance inst;
        inst.read(const_cast<char *>(input_path.c_str()));

        // Construct BDD
        KnapsackBDDConstructor bddCons(&inst);
        bdd = bddCons.generate_exact();
        // obj_coeffs = inst.obj_coeffs;

        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();

        // cout << "Original width: " << original_width << " - number of nodes: " << original_num_nodes << endl;

        // Reduce BDD
        BDDAlg::reduce(bdd);

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();

        // cout << "Reduced width: " << reduced_width << " - number of nodes: " << reduced_num_nodes << endl;

        // Update node weights
        bddCons.update_node_weights(bdd);

        // Reduce BDD
        BDDAlg::reduce(bdd);

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();

        // cout << "Reduced-2 width: " << reduced_width << " - number of nodes: " << reduced_num_nodes << endl;

        // Update node weights
        bddCons.update_node_weights(bdd);

        //        bdd->print();
    }

    // --- Set Packing ---
    else if (problem_type == 2)
    {

        // read instance
        SetPackingInstance setpack(input_path.c_str());

        // create associated independent set instance
        IndepSetInst *inst = setpack.create_indepset_instance();

        // generate independent set BDD
        IndepSetBDDConstructor bddConstructor(inst, setpack.objs);
        bdd = bddConstructor.generate_exact();

        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();
    }

    // --- TSP ---
    else if (problem_type == 3)
    {
        // Read instance
        TSPInstance inst;
        inst.read(input_path.c_str());

        // Construct MDD
        const WallClock::time_point compilation_tsp_wall_begin = WallClock::now();
        clock_t compilation_tsp = clock();

        MDDTSPConstructor mddCons(&inst);
        MDD *mdd = mddCons.generate_exact();
        assert(mdd != NULL);

        compilation_tsp = clock() - compilation_tsp;
        compilation_wall_s = std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - compilation_tsp_wall_begin).count();

        // Generate frontier (timed region excludes final lexicographic sort)
        const WallClock::time_point pareto_tsp_wall_begin = WallClock::now();
        clock_t pareto_tsp_cpu = clock();

        // Solver-owned stats populated during frontier enumeration.
        EnumerationStats *enumeration_stats = new EnumerationStats;
        enumeration_stats->cpu_perf_enabled = (backend == BACKEND_CPU) && save_stats;
        ParetoFrontier *pareto_frontier = NULL;

        if (method == 1) { // Top-down
            if (backend == BACKEND_GPU) {
                string cuda_reason;
                pareto_frontier = BDDMultiObj::pareto_frontier_topdown_cuda(mdd, enumeration_stats, &cuda_reason, kernel_version);
                if (pareto_frontier == NULL) {
                    cout << "Error - GPU backend requested but top-down enumeration failed";
                    if (!cuda_reason.empty()) cout << ": " << cuda_reason;
                    cout << endl;
                    exit(1);
                }
            } else {
                pareto_frontier = BDDMultiObj::pareto_frontier_topdown(mdd, enumeration_stats, cpu_threads);
            }
        } else if (method == 3) { // Coupled
            if (backend == BACKEND_GPU) {
                string cuda_reason;
                pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset_cuda(mdd, enumeration_stats, &cuda_reason, kernel_version);
                if (pareto_frontier == NULL) {
                    cout << "Error - GPU backend requested but coupled enumeration failed";
                    if (!cuda_reason.empty()) cout << ": " << cuda_reason;
                    cout << endl;
                    exit(1);
                }
            } else {
                pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(mdd, enumeration_stats, cpu_threads);
            }
        } else {
            cout << "Error - method " << method << " not valid for TSP" << endl;
            exit(1);
        }
        pareto_tsp_cpu = clock() - pareto_tsp_cpu;
        pareto_wall_enumeration_s = std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - pareto_tsp_wall_begin).count();

        assert(pareto_frontier != NULL);
        pareto_frontier->sort_lexicographic_ascending();

        if (save_frontier)
        {
            string save_error;
            if (!write_frontier_gzip_csv(pareto_frontier, frontier_out_path, &save_error))
            {
                cout << "Error - failed to save frontier to '" << frontier_out_path << "'";
                if (!save_error.empty())
                {
                    cout << ": " << save_error;
                }
                cout << endl;
                exit(1);
            }
        }

        const double wall_total_end_to_end_s =
            std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - run_wall_begin).count();

        // Run-level summary assembled in main for reporting/output only.
        RunSummaryStats run_summary;
        run_summary.is_tsp_branch = true;
        run_summary.postprocess_sort_applied = true;
        run_summary.num_solutions = pareto_frontier->get_num_sols();
        run_summary.original_width = -1;
        run_summary.reduced_width = -1;
        run_summary.original_num_nodes = -1;
        run_summary.reduced_num_nodes = -1;
        run_summary.layer_coupling = enumeration_stats->layer_coupling;
        run_summary.dominance_filtered_total = enumeration_stats->dominance_filtered_total;
        run_summary.cpu_dominance_s = ((double)enumeration_stats->cpu_ticks_dominance) / CLOCKS_PER_SEC;
        run_summary.cpu_compile_s = ((double)compilation_tsp) / CLOCKS_PER_SEC;
        run_summary.cpu_enumeration_s = ((double)pareto_tsp_cpu) / CLOCKS_PER_SEC;
        run_summary.cpu_total_s = run_summary.cpu_compile_s + run_summary.cpu_enumeration_s;
        run_summary.wall_compile_s = compilation_wall_s;
        run_summary.wall_enumeration_s = pareto_wall_enumeration_s;
        run_summary.wall_total_end_to_end_s = wall_total_end_to_end_s;

        print_and_save_run_summary(options, enumeration_stats, run_summary, pareto_frontier);

        return 0;
    }
    else
    {
        cout << "Error - problem type not recognized" << endl;
        exit(1);
    }

    timers.end_timer(bdd_compilation_time);
    compilation_wall_s = std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - compilation_wall_begin).count();

    // cout << "\nBDD Info:\n";
    // cout << "\tOriginal width: " << original_width << endl;
    // cout << "\tOriginal number of nodes: " << original_num_nodes << endl;
    // cout << "\n\tReduced width: " << reduced_width << endl;
    // cout << "\tReduced number of nodes: " << reduced_num_nodes << endl;
    // cout << "\n\tBDD compilation total time: " << timers.get_time(bdd_compilation_time) << endl;

    // Initialize enumeration stats
    // Solver-owned stats populated during frontier enumeration.
    EnumerationStats *enumeration_stats = new EnumerationStats;
    enumeration_stats->cpu_perf_enabled = (backend == BACKEND_CPU) && save_stats;

    // Compute pareto frontier based on methodology
    // cout << "\n\nComputing pareto frontier..." << endl;
    ParetoFrontier *pareto_frontier = NULL;
    const WallClock::time_point pareto_wall_begin = WallClock::now();
    timers.start_timer(pareto_time);

    if (method == 1)
    {
        // -- Optimal BFS algorithm: top-down --
        if (backend == BACKEND_GPU)
        {
            string cuda_reason;
            pareto_frontier = BDDMultiObj::pareto_frontier_topdown_cuda(bdd, maximization, problem_type, dominance, enumeration_stats, &cuda_reason, kernel_version);
            if (pareto_frontier == NULL)
            {
                cout << "Error - GPU backend requested but top-down enumeration failed";
                if (!cuda_reason.empty())
                {
                    cout << ": " << cuda_reason;
                }
                cout << endl;
                exit(1);
            }
        }
        else
        {
            pareto_frontier = BDDMultiObj::pareto_frontier_topdown(bdd, maximization, problem_type, dominance, enumeration_stats, cpu_threads);
        }
    }
    else if (method == 2)
    {
        // -- Optimal BFS algorithm: bottom-up --
        if (backend == BACKEND_GPU)
        {
            cout << "Error - GPU backend is unsupported for method 2." << endl;
            exit(1);
        }
        pareto_frontier = BDDMultiObj::pareto_frontier_bottomup(bdd, maximization, problem_type, dominance, enumeration_stats, cpu_threads);
    }
    else if (method == 3)
    {
        // -- Dynamic layer cutset --
        if (backend == BACKEND_GPU)
        {
            cout << "Error - GPU backend is unsupported for method 3." << endl;
            exit(1);
        }
        pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(bdd, maximization, problem_type, dominance, enumeration_stats, cpu_threads);
    }
    else
    {
        cout << "Error - method not recognized" << endl;
        exit(1);
    }

    if (pareto_frontier == NULL)
    {
        cout << "\nError - pareto frontier not computed" << endl;
        exit(1);
    }
    timers.end_timer(pareto_time);
    pareto_wall_enumeration_s = std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - pareto_wall_begin).count();

    pareto_frontier->sort_lexicographic_ascending();

    if (save_frontier)
    {
        string save_error;
        if (!write_frontier_gzip_csv(pareto_frontier, frontier_out_path, &save_error))
        {
            cout << "Error - failed to save frontier to '" << frontier_out_path << "'";
            if (!save_error.empty())
            {
                cout << ": " << save_error;
            }
            cout << endl;
            exit(1);
        }
    }

    // cout << "\nPareto frontier: " << endl;
    // cout << "\tNumber of solutions: " << pareto_frontier->get_num_sols() << endl;
    // cout << "\n\tBDD time: " << timers.get_time(bdd_compilation_time) << endl;
    // cout << "\tPareto time: " << timers.get_time(pareto_time) << endl;
    // cout << "\tTotal time: " << (timers.get_time(bdd_compilation_time) + timers.get_time(pareto_time)) << endl;
    // cout << endl;

    // cout << "\n\nPareto frontier: " << endl;
    // pareto_frontier->print();
    // cout << endl;

    // // Statistics file
    // ofstream stats("stats.txt", ios::app);
    // stats << argv[1];
    // stats << "\t" << problem_type;
    // stats << "\t" << NOBJS;
    // stats << "\t" << method;
    // stats << "\t" << pareto_frontier->get_num_sols();
    // stats << "\t" << original_width;
    // stats << "\t" << original_num_nodes;
    // stats << "\t" << reduced_width;
    // stats << "\t" << reduced_num_nodes;
    // stats << "\t" << timers.get_time(bdd_compilation_time);
    // stats << "\t" << timers.get_time(pareto_time);
    // stats << "\t" << (timers.get_time(bdd_compilation_time) + timers.get_time(pareto_time));
    // stats << endl;
    // stats.close();

    const double wall_total_end_to_end_s =
        std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - run_wall_begin).count();

    // Run-level summary assembled in main for reporting/output only.
    RunSummaryStats run_summary;
    run_summary.is_tsp_branch = false;
    run_summary.postprocess_sort_applied = true;
    run_summary.num_solutions = pareto_frontier->get_num_sols();
    run_summary.original_width = original_width;
    run_summary.reduced_width = reduced_width;
    run_summary.original_num_nodes = original_num_nodes;
    run_summary.reduced_num_nodes = reduced_num_nodes;
    run_summary.layer_coupling = enumeration_stats->layer_coupling;
    run_summary.dominance_filtered_total = enumeration_stats->dominance_filtered_total;
    run_summary.cpu_dominance_s = ((double)enumeration_stats->cpu_ticks_dominance) / CLOCKS_PER_SEC;
    run_summary.cpu_compile_s = timers.get_time(bdd_compilation_time);
    run_summary.cpu_enumeration_s = timers.get_time(pareto_time);
    run_summary.cpu_total_s = run_summary.cpu_compile_s + run_summary.cpu_enumeration_s;
    run_summary.wall_compile_s = compilation_wall_s;
    run_summary.wall_enumeration_s = pareto_wall_enumeration_s;
    run_summary.wall_total_end_to_end_s = wall_total_end_to_end_s;

    print_and_save_run_summary(options, enumeration_stats, run_summary, pareto_frontier);

    return 0;
}
