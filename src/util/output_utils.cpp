#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include "output_utils.hpp"
#include "util.hpp"

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

static string json_escape(const string &value)
{
    string escaped;
    escaped.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i)
    {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        switch (c)
        {
        case '\"':
            escaped += "\\\"";
            break;
        case '\\':
            escaped += "\\\\";
            break;
        case '\b':
            escaped += "\\b";
            break;
        case '\f':
            escaped += "\\f";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            if (c < 0x20)
            {
                static const char hex[] = "0123456789abcdef";
                escaped += "\\u00";
                escaped += hex[(c >> 4) & 0x0F];
                escaped += hex[c & 0x0F];
            }
            else
            {
                escaped += static_cast<char>(c);
            }
            break;
        }
    }
    return escaped;
}

bool write_frontier_gzip_csv(const ParetoFrontier *frontier, const string &out_path, string *error)
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
            if (o > 0 && fputc(',', pipe) == EOF)
            {
                ok = false;
                break;
            }
            if (fprintf(pipe, "%d", frontier->sols[i + o]) < 0)
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

bool write_stats_jsonl(const string &out_path,
                       const CliOptions &opts,
                       const EnumerationStats *stats,
                       const RunSummaryStats &record,
                       string *error)
{
    ofstream out(out_path.c_str(), ios::out | ios::app);
    if (!out.is_open())
    {
        if (error != NULL)
        {
            *error = "could not open output path";
        }
        return false;
    }

    out << fixed << setprecision(6);
    const double cpu_dominance_s = stats != NULL ? ((double)stats->cpu_ticks_dominance) / CLOCKS_PER_SEC : record.cpu_dominance_s;
    const int layer_coupling = stats != NULL ? stats->layer_coupling : record.layer_coupling;
    const int dominance_filtered_total = stats != NULL ? stats->dominance_filtered_total : record.dominance_filtered_total;
    const long long work_candidates_total = stats != NULL ? stats->work_candidates_total : 0;
    const long long work_frontier_survivors_total = stats != NULL ? stats->work_frontier_survivors_total : 0;
    const long long work_frontier_peak_points = stats != NULL ? stats->work_frontier_peak_points : 0;
    const long long work_join_products_total = stats != NULL ? stats->work_join_products_total : 0;

    out << "{";
    out << "\"schema_version\":1,";
    out << "\"identity\":{";
    out << "\"input_path\":\"" << json_escape(opts.input_path) << "\",";
    out << "\"problem_type\":" << opts.problem_type << ",";
    out << "\"method\":" << opts.method << ",";
    out << "\"dominance\":" << opts.dominance << ",";
    out << "\"backend\":\"" << backend_to_string(opts.backend) << "\",";
    out << "\"cpu_threads\":" << opts.cpu_threads << ",";
    out << "\"kernel_version\":" << opts.kernel_version;
    out << "},";

    out << "\"outputs\":{";
    out << "\"num_solutions\":" << record.num_solutions << ",";
    out << "\"save_frontier\":" << (opts.save_frontier ? "true" : "false") << ",";
    out << "\"frontier_out_path\":\"" << json_escape(opts.frontier_out_path) << "\"";
    out << "},";

    out << "\"timing\":{";
    out << "\"cpu\":{";
    out << "\"cpu_compile_s\":" << record.cpu_compile_s << ",";
    out << "\"cpu_enumeration_s\":" << record.cpu_enumeration_s << ",";
    out << "\"cpu_total_s\":" << record.cpu_total_s << ",";
    out << "\"cpu_dominance_s\":" << cpu_dominance_s;
    out << "},";
    out << "\"wall\":{";
    out << "\"wall_compile_s\":" << record.wall_compile_s << ",";
    out << "\"wall_enumeration_s\":" << record.wall_enumeration_s << ",";
    out << "\"wall_total_end_to_end_s\":" << record.wall_total_end_to_end_s;
    out << "}";
    out << "},";

    out << "\"work\":{";
    out << "\"work_candidates_total\":" << work_candidates_total << ",";
    out << "\"work_frontier_survivors_total\":" << work_frontier_survivors_total << ",";
    out << "\"work_frontier_peak_points\":" << work_frontier_peak_points << ",";
    out << "\"work_join_products_total\":" << work_join_products_total;
    out << "},";

    out << "\"dominance\":{";
    out << "\"dominance_filtered_total\":" << dominance_filtered_total << ",";
    out << "\"cpu_ticks_dominance\":" << (stats != NULL ? stats->cpu_ticks_dominance : 0);
    out << "},";

    out << "\"structure\":{";
    out << "\"is_tsp_branch\":" << (record.is_tsp_branch ? "true" : "false") << ",";
    out << "\"postprocess_sort_applied\":" << (record.postprocess_sort_applied ? "true" : "false") << ",";
    out << "\"original_width\":" << record.original_width << ",";
    out << "\"reduced_width\":" << record.reduced_width << ",";
    out << "\"original_num_nodes\":" << record.original_num_nodes << ",";
    out << "\"reduced_num_nodes\":" << record.reduced_num_nodes << ",";
    out << "\"layer_coupling\":" << layer_coupling;
    out << "},";

    out << "\"perf\":{";
    out << "\"wall_expand_td_s\":" << (stats != NULL ? stats->wall_expand_td_s : 0.0) << ",";
    out << "\"wall_expand_bu_s\":" << (stats != NULL ? stats->wall_expand_bu_s : 0.0) << ",";
    out << "\"wall_recompute_td_s\":" << (stats != NULL ? stats->wall_recompute_td_s : 0.0) << ",";
    out << "\"wall_recompute_bu_s\":" << (stats != NULL ? stats->wall_recompute_bu_s : 0.0) << ",";
    out << "\"wall_dominance_s\":" << (stats != NULL ? stats->wall_dominance_s : 0.0) << ",";
    out << "\"wall_cutset_sort_s\":" << (stats != NULL ? stats->wall_cutset_sort_s : 0.0) << ",";
    out << "\"wall_cutset_convolution_s\":" << (stats != NULL ? stats->wall_cutset_convolution_s : 0.0) << ",";
    out << "\"wall_cutset_partial_merge_s\":" << (stats != NULL ? stats->wall_cutset_partial_merge_s : 0.0) << ",";
    out << "\"wall_pack_transfer_s\":" << (stats != NULL ? stats->wall_pack_transfer_s : 0.0) << ",";
    out << "\"wall_join_s\":" << (stats != NULL ? stats->wall_join_s : 0.0) << ",";
    out << "\"cpu_layers_td\":" << (stats != NULL ? stats->cpu_layers_td : 0) << ",";
    out << "\"cpu_layers_bu\":" << (stats != NULL ? stats->cpu_layers_bu : 0) << ",";
    out << "\"cpu_nodes_expanded\":" << (stats != NULL ? stats->cpu_nodes_expanded : 0) << ",";
    out << "\"cpu_cutset_size\":" << (stats != NULL ? stats->cpu_cutset_size : 0);
    out << "},";

    out << "\"status\":{";
    out << "\"status_state\":\"" << json_escape(record.status_state) << "\",";
    out << "\"status_error_message\":\"" << json_escape(record.status_error_message) << "\"";
    out << "}";
    out << "}\n";

    if (!out.good())
    {
        if (error != NULL)
        {
            *error = "failed while writing JSONL output";
        }
        return false;
    }
    return true;
}

void print_and_save_run_summary(const CliOptions &opts, 
                                const EnumerationStats *enumeration_stats, 
                                const RunSummaryStats &run_summary, 
                                const ParetoFrontier *pareto_frontier)
{
    if (run_summary.is_tsp_branch)
    {
        cout << pareto_frontier->get_num_sols() << endl;
        cout << run_summary.cpu_total_s << endl;
        cout << run_summary.cpu_compile_s;
        cout << "\t" << run_summary.cpu_enumeration_s;
        cout << "\t" << run_summary.wall_compile_s;
        cout << "\t" << run_summary.wall_enumeration_s;
        cout << "\t" << run_summary.wall_total_end_to_end_s;
        cout << endl;
    }
    else
    {
        cout << pareto_frontier->get_num_sols() << endl;
        cout << run_summary.cpu_total_s << endl;

        cout << opts.method;
        cout << "\t" << opts.dominance;
        cout << "\t" << run_summary.original_width;
        cout << "\t" << run_summary.reduced_width;
        cout << "\t" << run_summary.original_num_nodes;
        cout << "\t" << run_summary.reduced_num_nodes;
        cout << "\t" << run_summary.cpu_compile_s;
        cout << "\t" << run_summary.cpu_enumeration_s;
        cout << "\t" << enumeration_stats->layer_coupling;
        cout << "\t" << enumeration_stats->dominance_filtered_total;
        cout << "\t" << run_summary.cpu_dominance_s;
        cout << "\t" << run_summary.wall_compile_s;
        cout << "\t" << run_summary.wall_enumeration_s;
        cout << "\t" << run_summary.wall_total_end_to_end_s;
        cout << endl;
    }

    if (opts.save_stats)
    {
        string stats_error;
        if (!write_stats_jsonl(opts.stats_out_path, opts, enumeration_stats, run_summary, &stats_error))
        {
            cerr << "Error - failed to save stats to '" << opts.stats_out_path << "'";
            if (!stats_error.empty())
            {
                cerr << ": " << stats_error;
            }
            cerr << '\n';
            exit(1);
        }
    }
}
