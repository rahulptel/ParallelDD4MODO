// --------------------------------------------------
// Multiobjective
// --------------------------------------------------

// General includes
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <cerrno>
#include <climits>
#include <chrono>
#include <iomanip>

#include "bdd/bdd.hpp"
#include "bdd/bdd_alg.hpp"
#include "bdd/bdd_multiobj.hpp"
#include "util/stats.hpp"
#include "util/util.hpp"
#include "bdd/pareto_frontier.hpp"

// Knapsack includes
#include "instances/knapsack_instance.hpp"
#include "bdd/knapsack_bdd.hpp"

// Set packing / Independent set includes
#include "instances/indepset_instance.hpp"
#include "instances/setpacking_instance.hpp"
#include "bdd/indepset_bdd.hpp"

// Set covering includes
#include "instances/setcovering_instance.hpp"
#include "bdd/setcovering_bdd.hpp"

// TSP instance
#include "instances/tsp_instance.hpp"
#include "mdd/tsp_mdd.hpp"

using namespace std;

static string shell_single_quote(const string &value)
{
    string quoted = "'";
    for (char c : value)
    {
        if (c == '\'')
        {
            quoted += "'\"'\"'";
        }
        else
        {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

static string derive_default_frontier_path(const string &input_path)
{
    string base = input_path;
    size_t slash_pos = base.find_last_of("/\\");
    if (slash_pos != string::npos)
    {
        base = base.substr(slash_pos + 1);
    }

    size_t dot_pos = base.find_last_of('.');
    if (dot_pos != string::npos)
    {
        base = base.substr(0, dot_pos);
    }

    if (base.empty())
    {
        base = "frontier";
    }

    return base + ".frontier.csv.gz";
}

static bool parse_positive_int(const string &value, int *out_value)
{
    if (value.empty())
    {
        return false;
    }

    errno = 0;
    char *end_ptr = NULL;
    long parsed = strtol(value.c_str(), &end_ptr, 10);
    if (errno != 0 || end_ptr == value.c_str() || *end_ptr != '\0')
    {
        return false;
    }
    if (parsed <= 0 || parsed > INT_MAX)
    {
        return false;
    }

    if (out_value != NULL)
    {
        *out_value = static_cast<int>(parsed);
    }
    return true;
}

static bool write_frontier_gzip_csv(const ParetoFrontier *frontier, const int problem_type, const string &out_path, string *error)
{
    if (frontier == NULL)
    {
        if (error != NULL)
        {
            *error = "frontier is null";
        }
        return false;
    }

    if (out_path.empty())
    {
        if (error != NULL)
        {
            *error = "output path is empty";
        }
        return false;
    }

    if (frontier->sols.size() % NOBJS != 0)
    {
        if (error != NULL)
        {
            *error = "frontier has invalid dimension";
        }
        return false;
    }

    const string command = "gzip -c > " + shell_single_quote(out_path);
    FILE *pipe = popen(command.c_str(), "w");
    if (pipe == NULL)
    {
        if (error != NULL)
        {
            *error = "could not launch gzip";
        }
        return false;
    }

    bool ok = true;
    for (size_t i = 0; i < frontier->sols.size() && ok; i += NOBJS)
    {
        for (int o = 0; o < NOBJS; ++o)
        {
            ObjType value = frontier->sols[i + o];
            if (problem_type == 3)
            {
                value = -value;
            }

            if (o > 0 && fputc(',', pipe) == EOF)
            {
                ok = false;
                break;
            }
            if (fprintf(pipe, "%d", value) < 0)
            {
                ok = false;
                break;
            }
        }
        if (ok && fputc('\n', pipe) == EOF)
        {
            ok = false;
        }
    }

    int close_status = pclose(pipe);
    if (!ok)
    {
        if (error != NULL)
        {
            *error = "failed while writing compressed frontier";
        }
        return false;
    }

    if (close_status != 0)
    {
        if (error != NULL)
        {
            *error = "gzip exited with non-zero status";
        }
        return false;
    }

    return true;
}

static void print_perf_log(const string &input_path,
                           const int problem_type,
                           const int method,
                           const string &backend_name,
                           const int cpu_threads,
                           const MultiObjectiveStats *stats,
                           const double compile_wall_s,
                           const double enum_wall_s,
                           const double total_wall_s,
                           const double compile_cpu_s,
                           const double enum_cpu_s,
                           const bool postprocess_sort_applied)
{
    cerr << fixed << setprecision(6);
    cerr << "[perf] input=" << input_path
         << " problem_type=" << problem_type
         << " method=" << method
         << " backend=" << backend_name
         << " cpu_threads=" << cpu_threads << '\n';
    cerr << "[perf] wall_s compile=" << compile_wall_s
         << " enum=" << enum_wall_s
         << " total=" << total_wall_s
         << " postprocess_sort=" << (postprocess_sort_applied ? "applied" : "skipped") << '\n';
    cerr << "[perf] cpu_s compile=" << compile_cpu_s
         << " enum=" << enum_cpu_s << '\n';
    if (stats != NULL)
    {
        cerr << "[perf] phases_s expand_td=" << stats->cpu_expand_td_wall_s
             << " expand_bu=" << stats->cpu_expand_bu_wall_s
             << " recompute_td=" << stats->cpu_recompute_td_wall_s
             << " recompute_bu=" << stats->cpu_recompute_bu_wall_s
             << " dominance=" << stats->cpu_dominance_wall_s
             << " cutset_sort=" << stats->cpu_cutset_sort_wall_s
             << " cutset_convolution=" << stats->cpu_cutset_convolution_wall_s
             << " cutset_partial_merge=" << stats->cpu_cutset_partial_merge_wall_s << '\n';
        cerr << "[perf] counters layers_td=" << stats->cpu_layers_td
             << " layers_bu=" << stats->cpu_layers_bu
             << " nodes_expanded=" << stats->cpu_nodes_expanded
             << " cutset_size=" << stats->cpu_cutset_size
             << " dominance_filtered=" << stats->pareto_dominance_filtered
             << " dominance_cpu_s=" << ((double)stats->pareto_dominance_time) / CLOCKS_PER_SEC << '\n';
    }
}

//
// Main function
//
int main(int argc, char *argv[])
{
    enum Backend
    {
        BACKEND_CPU = 0,
        BACKEND_GPU = 1
    };

    auto print_usage = []()
    {
        cout << '\n';
        cout << "Usage: multiobj_nobjs<NUM_OBJS> [input file] [problem type] [preprocess?] [method] [appr-S] [appr-T] [dominance] [options]\n";

        cout << "\n\twhere:";

        cout << "\n";
        cout << "\t\tproblem_type = 1: knapsack\n";
        cout << "\t\tproblem_type = 2: set packing\n";
        cout << "\t\tproblem_type = 3: set covering\n";
        cout << "\t\tproblem_type = 4: TSP\n";

        cout << "\n";
        cout << "\t\tpreprocess = 0: do not preprocess instance\n";
        cout << "\t\tpreprocess = 1: preprocess input to minimize BDD size\n";

        cout << "\n";
        cout << "\t\tmethod = 1: top-down BFS\n";
        cout << "\t\tmethod = 2: bottom-up BFS\n";
        cout << "\t\tmethod = 3: dynamic layer cutset\n";

        cout << "\n";
        cout << "\t\tapprox = n m: approximate n-sized S set and m-sized T set (n=0 if disabled)\n";

        cout << "\n";
        cout << "\t\tdominance = 0:  disable state dominance\n";
        cout << "\t\tdominance = 1:  state dominance strategy 1\n";

        cout << "\n";
        cout << "\t\tNamed backend options:\n";
        cout << "\t\t\t--backend cpu|gpu\n";
        cout << "\t\t\t--cpu-threads <N>   (cpu only)\n";
        cout << "\t\t\t--kernel <K>        (gpu only, K in {1,2,3})\n";
        cout << "\t\tShorthand backend options:\n";
        cout << "\t\t\tcpu [N]\n";
        cout << "\t\t\tgpu [K]\n";
        cout << "\t\tbackend omitted defaults to cpu\n";
        cout << "\t\tcpu threads default to OMP_NUM_THREADS if valid, otherwise 1\n";

        cout << "\n";
        cout << "\t\tkernel = 1: one block per node\n";
        cout << "\t\tkernel = 2: fixed number of blocks per node (2D grid)\n";
        cout << "\t\tkernel = 3: dynamic blocks per node with binary-search destination lookup (1D grid)\n";
        cout << "\t\tkernel omitted with backend=gpu: defaults by problem type\n";
        cout << "\t\t\tproblem_type=1 (knapsack): 1\n";
        cout << "\t\t\tproblem_type=2 (set packing): 2\n";
        cout << "\t\t\tproblem_type=3 (set covering): 1\n";
        cout << "\t\t\tproblem_type=4 (TSP): 3\n";

        cout << "\n";
        cout << "\t\t--save-frontier: save Pareto frontier to <input_stem>.frontier.csv.gz\n";
        cout << "\t\t--frontier-out <path>: save Pareto frontier to explicit gzip CSV path\n";
        cout << "\t\t--perf-log: print aggregated CPU performance diagnostics to stderr\n";
        cout << "\t\toptional arguments can be provided in any order\n";

        cout << endl;
    };

    if (argc < 8)
    {
        print_usage();
        exit(1);
    }

    // Read input
    int problem_type = atoi(argv[2]);
    bool preprocess = (argv[3][0] == '1');
    int method = atoi(argv[4]);
    bool maximization = true;
    int approx_S = atoi(argv[5]);
    int approx_T = atoi(argv[6]);
    int dominance = atoi(argv[7]);

    Backend backend = BACKEND_CPU;
    bool backend_set = false;
    bool backend_from_named = false;
    bool backend_from_shorthand = false;
    int kernel_version = -1;
    bool kernel_version_set = false;
    int cpu_threads = 1;
    bool cpu_threads_set = false;

    bool save_frontier = false;
    string frontier_out_path;
    bool perf_log = false;

    for (int i = 8; i < argc; ++i)
    {
        string token(argv[i]);
        if (token == "--backend")
        {
            if (backend_from_shorthand)
            {
                cout << "Error - cannot mix --backend with shorthand backend token." << endl;
                print_usage();
                exit(1);
            }
            if (backend_set)
            {
                cout << "Error - backend provided multiple times." << endl;
                print_usage();
                exit(1);
            }
            if (i + 1 >= argc)
            {
                cout << "Error - --backend requires a value (cpu or gpu)." << endl;
                print_usage();
                exit(1);
            }
            string backend_token(argv[++i]);
            if (backend_token == "cpu")
            {
                backend = BACKEND_CPU;
            }
            else if (backend_token == "gpu")
            {
                backend = BACKEND_GPU;
            }
            else if (backend_token == "cuda")
            {
                cout << "Error - backend token 'cuda' is unsupported; use 'gpu'." << endl;
                exit(1);
            }
            else
            {
                cout << "Error - invalid backend '" << backend_token << "'. Use cpu or gpu." << endl;
                print_usage();
                exit(1);
            }
            backend_set = true;
            backend_from_named = true;
        }
        else if (token == "--cpu-threads")
        {
            if (cpu_threads_set)
            {
                cout << "Error - cpu thread count provided multiple times." << endl;
                print_usage();
                exit(1);
            }
            if (i + 1 >= argc)
            {
                cout << "Error - --cpu-threads requires a positive integer." << endl;
                print_usage();
                exit(1);
            }
            string value(argv[++i]);
            if (!parse_positive_int(value, &cpu_threads))
            {
                cout << "Error - invalid --cpu-threads value '" << value << "' (expected positive integer)." << endl;
                exit(1);
            }
            cpu_threads_set = true;
        }
        else if (token == "--kernel")
        {
            if (kernel_version_set)
            {
                cout << "Error - kernel provided multiple times." << endl;
                print_usage();
                exit(1);
            }
            if (i + 1 >= argc)
            {
                cout << "Error - --kernel requires a value in {1,2,3}." << endl;
                print_usage();
                exit(1);
            }
            string value(argv[++i]);
            if (!parse_positive_int(value, &kernel_version) || kernel_version < 1 || kernel_version > 3)
            {
                cout << "Error - invalid --kernel value '" << value << "' (expected 1, 2, or 3)." << endl;
                exit(1);
            }
            kernel_version_set = true;
        }
        else if (token == "cpu" || token == "gpu")
        {
            if (backend_from_named)
            {
                cout << "Error - cannot mix shorthand backend token with --backend." << endl;
                print_usage();
                exit(1);
            }
            if (backend_set)
            {
                cout << "Error - backend provided multiple times." << endl;
                print_usage();
                exit(1);
            }

            backend = (token == "cpu" ? BACKEND_CPU : BACKEND_GPU);
            backend_set = true;
            backend_from_shorthand = true;

            if (i + 1 < argc)
            {
                string next_token(argv[i + 1]);
                errno = 0;
                char *end_ptr = NULL;
                long parsed_raw = strtol(next_token.c_str(), &end_ptr, 10);
                const bool next_is_integer = (errno == 0 && end_ptr != next_token.c_str() && *end_ptr == '\0');
                if (next_is_integer)
                {
                    if (token == "cpu")
                    {
                        if (cpu_threads_set)
                        {
                            cout << "Error - cpu threads provided multiple times." << endl;
                            print_usage();
                            exit(1);
                        }
                        if (parsed_raw <= 0 || parsed_raw > INT_MAX)
                        {
                            cout << "Error - invalid cpu shorthand thread count '" << next_token << "' (expected positive integer)." << endl;
                            exit(1);
                        }
                        cpu_threads = static_cast<int>(parsed_raw);
                        cpu_threads_set = true;
                    }
                    else
                    {
                        if (kernel_version_set)
                        {
                            cout << "Error - kernel provided multiple times." << endl;
                            print_usage();
                            exit(1);
                        }
                        if (parsed_raw < 1 || parsed_raw > 3)
                        {
                            cout << "Error - invalid gpu shorthand kernel '" << next_token << "' (expected 1, 2, or 3)." << endl;
                            exit(1);
                        }
                        kernel_version = static_cast<int>(parsed_raw);
                        kernel_version_set = true;
                    }
                    ++i;
                }
            }
        }
        else if (token == "cuda")
        {
            cout << "Error - backend token 'cuda' is unsupported; use 'gpu'." << endl;
            exit(1);
        }
        else if (token == "--save-frontier")
        {
            save_frontier = true;
        }
        else if (token == "--frontier-out")
        {
            if (i + 1 >= argc)
            {
                cout << "Error - --frontier-out requires a file path." << endl;
                print_usage();
                exit(1);
            }
            frontier_out_path = argv[++i];
            if (frontier_out_path.empty())
            {
                cout << "Error - --frontier-out path cannot be empty." << endl;
                exit(1);
            }
            save_frontier = true;
        }
        else if (token == "--perf-log")
        {
            perf_log = true;
        }
        else
        {
            int parsed_numeric = 0;
            if (parse_positive_int(token, &parsed_numeric))
            {
                cout << "Error - positional numeric argument '" << token << "' must immediately follow shorthand backend token cpu|gpu." << endl;
            }
            else
            {
                cout << "Error - unrecognized optional argument '" << token << "'." << endl;
            }
            print_usage();
            exit(1);
        }
    }

    if (save_frontier && frontier_out_path.empty())
    {
        frontier_out_path = derive_default_frontier_path(argv[1]);
    }

    if (backend == BACKEND_GPU && cpu_threads_set)
    {
        cout << "Error - cpu thread options are not valid with backend=gpu." << endl;
        exit(1);
    }

    if (backend == BACKEND_CPU && kernel_version_set)
    {
        cout << "Error - --kernel is only valid with backend=gpu." << endl;
        exit(1);
    }

    if (backend == BACKEND_GPU && !kernel_version_set)
    {
        if (problem_type == 1 || problem_type == 3)
        {
            kernel_version = 1;
        }
        else if (problem_type == 2)
        {
            kernel_version = 2;
        }
        else if (problem_type == 4)
        {
            kernel_version = 3;
        }
        else
        {
            cout << "Error - problem type not recognized" << endl;
            exit(1);
        }
    }
    else if (backend == BACKEND_CPU && !cpu_threads_set)
    {
        const char *env_threads = getenv("OMP_NUM_THREADS");
        int parsed_env_threads = 0;
        if (env_threads != NULL && parse_positive_int(string(env_threads), &parsed_env_threads))
        {
            cpu_threads = parsed_env_threads;
        }
        else
        {
            cpu_threads = 1;
        }
    }

    if (backend == BACKEND_GPU && method != 1 && method != 3)
    {
        cout << "Error - GPU backend is unsupported for method " << method << "." << endl;
        exit(1);
    }

    typedef std::chrono::steady_clock WallClock;
    const WallClock::time_point run_wall_begin = WallClock::now();
    double compilation_wall_s = 0.0;
    double pareto_enum_wall_s = 0.0;

    // For statistical analysis
    Stats timers;
    int bdd_compilation_time = timers.register_name("BDD compilation time");
    int pareto_time = timers.register_name("BDD pareto time");
    int approx_time = timers.register_name("BDD approximation time");
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
        inst.read(argv[1]);

        // if (preprocess) {
        //     // Reorder variables
        //     inst.reorder_coefficients();
        // }

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

        // Compute approximation
        if (approx_S != 0)
        {
            timers.start_timer(approx_time);
            // BDDMultiObj::approximate_pareto_frontier_bottomup(bdd, approx_S, approx_T);
            // BDDMultiObj::approximate_pareto_frontier_topdown(bdd, approx_S, approx_T);
            // BDDMultiObj::approximate_pareto_frontier_topdown_dominance(bdd, approx_S, approx_T);
            timers.end_timer(approx_time);
        }

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
        SetPackingInstance setpack(argv[1]);

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

    // --- Set Covering ---
    else if (problem_type == 3)
    {
        // set objective sense
        maximization = false;

        // read instance
        SetCoveringInstance setcover(argv[1]);

        // preprocess
        if (preprocess)
        {
            setcover.minimize_bandwidth();
        }

        // create BDD
        SetCoveringBDDConstructor bddConstructor(&setcover, setcover.objs);
        bdd = bddConstructor.generate_exact();

        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();

        // Reduce BDD
        // BDDAlg::reduce(bdd);

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();

        // --- TSP ---
    }
    else if (problem_type == 4)
    {
        // Read instance
        TSPInstance inst;
        inst.read(argv[1]);

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

        MultiObjectiveStats *statsMultiObj = new MultiObjectiveStats;
        statsMultiObj->cpu_perf_enabled = (perf_log && backend == BACKEND_CPU);
        ParetoFrontier *pareto_frontier = NULL;

        if (method == 1) { // Top-down
            if (backend == BACKEND_GPU) {
                string cuda_reason;
                pareto_frontier = BDDMultiObj::pareto_frontier_topdown_cuda(mdd, statsMultiObj, &cuda_reason, kernel_version);
                if (pareto_frontier == NULL) {
                    cout << "Error - GPU backend requested but top-down enumeration failed";
                    if (!cuda_reason.empty()) cout << ": " << cuda_reason;
                    cout << endl;
                    exit(1);
                }
            } else {
                pareto_frontier = BDDMultiObj::pareto_frontier_topdown(mdd, statsMultiObj, cpu_threads);
            }
        } else if (method == 3) { // Coupled
            if (backend == BACKEND_GPU) {
                string cuda_reason;
                pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset_cuda(mdd, statsMultiObj, &cuda_reason, kernel_version);
                if (pareto_frontier == NULL) {
                    cout << "Error - GPU backend requested but coupled enumeration failed";
                    if (!cuda_reason.empty()) cout << ": " << cuda_reason;
                    cout << endl;
                    exit(1);
                }
            } else {
                pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(mdd, statsMultiObj, cpu_threads);
            }
        } else {
            cout << "Error - method " << method << " not valid for TSP" << endl;
            exit(1);
        }
        pareto_tsp_cpu = clock() - pareto_tsp_cpu;
        pareto_enum_wall_s = std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - pareto_tsp_wall_begin).count();

        assert(pareto_frontier != NULL);
        pareto_frontier->sort_lexicographic_ascending();

        if (save_frontier)
        {
            string save_error;
            if (!write_frontier_gzip_csv(pareto_frontier, problem_type, frontier_out_path, &save_error))
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

        const double total_wall_s_end_to_end =
            std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - run_wall_begin).count();

        cout << pareto_frontier->get_num_sols() << endl;
        cout << (double)(compilation_tsp + pareto_tsp_cpu) / CLOCKS_PER_SEC << endl;
        cout << (double)compilation_tsp / CLOCKS_PER_SEC;
        cout << "\t" << pareto_tsp_cpu / CLOCKS_PER_SEC;
        cout << "\t" << compilation_wall_s;
        cout << "\t" << pareto_enum_wall_s;
        cout << "\t" << total_wall_s_end_to_end;
        cout << endl;

        if (perf_log)
        {
            const double total_wall_s = total_wall_s_end_to_end;
            const string backend_name = (backend == BACKEND_GPU ? "gpu" : "cpu");
            print_perf_log(argv[1],
                           problem_type,
                           method,
                           backend_name,
                           cpu_threads,
                           statsMultiObj,
                           compilation_wall_s,
                           pareto_enum_wall_s,
                           total_wall_s,
                           ((double)compilation_tsp) / CLOCKS_PER_SEC,
                           ((double)pareto_tsp_cpu) / CLOCKS_PER_SEC,
                           true);
        }

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

    // Initialize multiobjective stats
    MultiObjectiveStats *statsMultiObj = new MultiObjectiveStats;
    statsMultiObj->cpu_perf_enabled = (perf_log && backend == BACKEND_CPU);

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
            pareto_frontier = BDDMultiObj::pareto_frontier_topdown_cuda(bdd, maximization, problem_type, dominance, statsMultiObj, &cuda_reason, kernel_version);
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
            pareto_frontier = BDDMultiObj::pareto_frontier_topdown(bdd, maximization, problem_type, dominance, statsMultiObj, cpu_threads);
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
        pareto_frontier = BDDMultiObj::pareto_frontier_bottomup(bdd, maximization, problem_type, dominance, statsMultiObj, cpu_threads);
    }
    else if (method == 3)
    {
        // -- Dynamic layer cutset --
        if (backend == BACKEND_GPU)
        {
            cout << "Error - GPU backend is unsupported for method 3." << endl;
            exit(1);
        }
        pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(bdd, maximization, problem_type, dominance, statsMultiObj, cpu_threads);
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
    pareto_enum_wall_s = std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - pareto_wall_begin).count();

    pareto_frontier->sort_lexicographic_ascending();

    if (save_frontier)
    {
        string save_error;
        if (!write_frontier_gzip_csv(pareto_frontier, problem_type, frontier_out_path, &save_error))
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

    double total_time = (timers.get_time(bdd_compilation_time) + timers.get_time(approx_time) + timers.get_time(pareto_time));
    (void)total_time;

    // cout << "\nPareto frontier: " << endl;
    // cout << "\tNumber of solutions: " << pareto_frontier->get_num_sols() << endl;
    // cout << "\n\tBDD time: " << timers.get_time(bdd_compilation_time) << endl;
    // cout << "\tApproximation filtering time: " << timers.get_time(approx_time) << endl;
    // cout << "\tPareto time: " << timers.get_time(pareto_time) << endl;
    // cout << "\tTotal time: " << total_time << endl;
    // cout << endl;

    // cout << "\n\nPareto frontier: " << endl;
    // pareto_frontier->print();
    // cout << endl;

    // // Statistics file
    // ofstream stats("stats.txt", ios::app);
    // stats << argv[1];
    // stats << "\t" << problem_type;
    // stats << "\t" << NOBJS;
    // stats << "\t" << preprocess;
    // stats << "\t" << method;
    // stats << "\t" << approx_S;
    // stats << "\t" << approx_T;
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

    const double total_wall_s_end_to_end =
        std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - run_wall_begin).count();

    cout << pareto_frontier->get_num_sols() << endl;
    cout << (timers.get_time(bdd_compilation_time) + timers.get_time(pareto_time)) << endl;

    cout << method;
    cout << "\t" << dominance;
    cout << "\t" << original_width;
    cout << "\t" << reduced_width;
    cout << "\t" << original_num_nodes;
    cout << "\t" << reduced_num_nodes;
    cout << "\t" << timers.get_time(bdd_compilation_time);
    cout << "\t" << timers.get_time(pareto_time);
    cout << "\t" << statsMultiObj->layer_coupling;
    cout << "\t" << statsMultiObj->pareto_dominance_filtered;
    cout << "\t" << ((double)statsMultiObj->pareto_dominance_time) / CLOCKS_PER_SEC;
    cout << "\t" << compilation_wall_s;
    cout << "\t" << pareto_enum_wall_s;
    cout << "\t" << total_wall_s_end_to_end;
    cout << endl;

    if (perf_log)
    {
        const double total_wall_s = total_wall_s_end_to_end;
        const string backend_name = (backend == BACKEND_GPU ? "gpu" : "cpu");
        print_perf_log(argv[1],
                       problem_type,
                       method,
                       backend_name,
                       cpu_threads,
                       statsMultiObj,
                       compilation_wall_s,
                       pareto_enum_wall_s,
                       total_wall_s,
                       timers.get_time(bdd_compilation_time),
                       timers.get_time(pareto_time),
                       true);
    }

    return 0;
}
