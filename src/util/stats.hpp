/*
 * -------------------------------------------------------------------
 * Statistic routines for profiling
 * -------------------------------------------------------------------
 */

#ifndef STATS_HPP_
#define STATS_HPP_

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <vector>
#include <string>

#define UNITIALIZED_STAT    -1    // initial value stat field receives
#define INITIAL_STAT_SIZE  100    // initial number of statistics considered


using namespace std;

//
// Enumeration stats populated by BDD/CUDA during frontier enumeration.
// This struct intentionally excludes run-level reporting fields.
//
struct EnumerationStats {
    // Time spent in pareto dominance filtering
    std::clock_t cpu_ticks_dominance;
    // Solutions filtered by pareto dominance
    int dominance_filtered_total;
    // Layer where coupling happened
    int layer_coupling;
    // Enable lightweight CPU performance aggregation
    bool cpu_perf_enabled;
    // Aggregated wall times (seconds) for CPU phases
    double wall_expand_td_s;
    double wall_expand_bu_s;
    double wall_recompute_td_s;
    double wall_recompute_bu_s;
    double wall_dominance_s;
    double wall_cutset_sort_s;
    double wall_cutset_convolution_s;
    double wall_cutset_partial_merge_s;
    // Aggregated wall times (seconds) for GPU packing and join phases
    double wall_pack_transfer_s;
    double wall_join_s;
    // Method-agnostic work counters
    long long work_candidates_total;
    long long work_frontier_survivors_total;
    long long work_frontier_peak_points;
    long long work_join_products_total;
    // Aggregated counters for CPU phases
    int cpu_layers_td;
    int cpu_layers_bu;
    long long cpu_nodes_expanded;
    int cpu_cutset_size;

    // Constructor
    EnumerationStats() 
    : cpu_ticks_dominance(0),
      dominance_filtered_total(0),
      layer_coupling(0),
      cpu_perf_enabled(false),
      wall_expand_td_s(0.0),
      wall_expand_bu_s(0.0),
      wall_recompute_td_s(0.0),
      wall_recompute_bu_s(0.0),
      wall_dominance_s(0.0),
      wall_cutset_sort_s(0.0),
      wall_cutset_convolution_s(0.0),
      wall_cutset_partial_merge_s(0.0),
      wall_pack_transfer_s(0.0),
      wall_join_s(0.0),
      work_candidates_total(0),
      work_frontier_survivors_total(0),
      work_frontier_peak_points(0),
      work_join_products_total(0),
      cpu_layers_td(0),
      cpu_layers_bu(0),
      cpu_nodes_expanded(0),
      cpu_cutset_size(0)
    { }
};

// Backward-compatible alias: existing code can keep using MultiObjectiveStats.
using MultiObjectiveStats = EnumerationStats;

struct RunSummaryStats
{
    bool is_tsp_branch;
    bool postprocess_sort_applied;
    int num_solutions;
    long original_width;
    long reduced_width;
    long original_num_nodes;
    long reduced_num_nodes;
    int layer_coupling;
    int dominance_filtered_total;
    double cpu_dominance_s;
    double cpu_compile_s;
    double cpu_enumeration_s;
    double cpu_total_s;
    double wall_compile_s;
    double wall_enumeration_s;
    double wall_total_end_to_end_s;
    std::string status_state;
    std::string status_error_message;

    RunSummaryStats()
        : is_tsp_branch(false),
          postprocess_sort_applied(true),
          num_solutions(0),
          original_width(-1),
          reduced_width(-1),
          original_num_nodes(-1),
          reduced_num_nodes(-1),
          layer_coupling(0),
          dominance_filtered_total(0),
          cpu_dominance_s(0.0),
          cpu_compile_s(0.0),
          cpu_enumeration_s(0.0),
          cpu_total_s(0.0),
          wall_compile_s(0.0),
          wall_enumeration_s(0.0),
          wall_total_end_to_end_s(0.0),
          status_state("ok"),
          status_error_message("")
    {
    }
};
/**
 * Struct for C-string comparison
 */
struct ltstr {
    bool operator()(const char* s1, const char* s2) const {
        return( strcmp(s1, s2) < 0 );
    }
};

/**
 * Struct for statistic value
 */
struct data_t {
    int id;
    data_t(): id(UNITIALIZED_STAT) { }
};



/**
 * Class to collect general statistics on the code
 */
class TimeStats {

public:

    /** Constructor: Just reserve memory */
    TimeStats() {
        timer_start.reserve(INITIAL_STAT_SIZE);
        value.reserve(INITIAL_STAT_SIZE);
    }

    /** Register a new statistic. The initial value is 0 by default */
    int register_name(const char* name, long int initial_value = 0);

    /** Start timer for a time statistic */
    void start_timer(const char* name);

    /** End timer for a time statistic, accumulating the result (in seconds) */
    void end_timer(const char* name);

    /** Start timer (by id) for a time statistic */
    void start_timer(int id);

    /** End timer (by id) for a time statistic, accumulating the result (in seconds) */
    void end_timer(int id);

    /** Add value for a numerical statistic by name. Default value to add is 1. */
    void add_value(const char* name, long int val=1);

    /** Add value for a numerical statistic by id. Default value to add is 1. */
    void add_value(int id, long int val=1);

    /** Get value for a statistic*/
    long int get_value(const char* name);

    /** Get value for a statistic*/
    long int get_value(int id);

    /** Return value (by name) interpreted as time */
    double get_time(const char* name);

    /** Return value (by id) interpreted as time */
    double get_time(int id);

    /** Return value (by name) interpreted as time, taking current time clock for measure */
    double get_current_time(const char* name);

    /** Return value (by id) interpreted as time, taking current time clock for measure */
    double get_current_time(int id);

    /** Return id of name */
    int get_id(const char* name);

    /** Print stats */
    void print();

private:

    map<const char*, data_t, ltstr>  name_to_id;     /**< map from name to stat identifier */
    vector<std::clock_t>                  timer_start;    /**< timer start for statistic */
    vector<long int>                 value;          /**< statistic value */
};


/**
 * -------------------------------------------------------------
 * Inline Implementations
 * -------------------------------------------------------------
 */


/**
 * Register a new statistic
 * */
inline int TimeStats::register_name(const char* name, long int initial_value) {
    name_to_id[name].id = value.size();
    value.push_back(initial_value);
    timer_start.push_back(clock());
    return value.size()-1;
}


/**
 * Start the timer for a statistic
 */
inline void TimeStats::start_timer(const char* name) {
    start_timer(get_id(name));
}


/**
 * Start the timer (by id) for a statistic
 */
inline void TimeStats::start_timer(int id) {
    assert( id >= 0 && id < (int)value.size() );
    timer_start[id] = clock();
}

/**
 * End the timer for a statistic, accumulating the result (in seconds)
 */
inline void TimeStats::end_timer(const char* name) {
    end_timer(get_id(name));
}

/**
 * End the timer for a statistic (by id), accumulating the result (in seconds)
 */
inline void TimeStats::end_timer(int id) {
    assert( id >= 0 && id < (int)value.size() );
    value[id] += clock() - timer_start[id];
}

/**
 * Add value for a numerical statistic. Default value to add is 1.
 */
inline void TimeStats::add_value(const char* name, long int val) {
    add_value(get_id(name), val);
}

/**
 * Add value by id
 */
inline void TimeStats::add_value(int id, long int val) {
    assert( id >= 0 && id < (int)value.size() );
    value[id] += val;
}


/**
 * Get value for a statistic
 */
inline long int TimeStats::get_value(const char* name) {
    return( get_value(get_id(name)) );
}


/**
 * Get value for a statistic
 */
inline long int TimeStats::get_value(int id) {
    assert( id >= 0 && id < (int)value.size() );
    return( value[id] );
}



/**
 * Get value for a statistic interpreted as time
 */
inline double TimeStats::get_time(const char* name) {
    return get_time(get_id(name));
}


/**
 * Get value for a statistic interpreted as time, using current time as measure
 */
inline double TimeStats::get_current_time(const char* name) {
    return get_current_time(get_id(name));
}


/**
 * Get value for a statistic interpreted as time, using current time as measure
 */
inline double TimeStats::get_current_time(int id) {
    assert( id >= 0 && id < (int)value.size() );
    return ((double)(clock() - timer_start[id])) / (double)CLOCKS_PER_SEC;
}


/**
 * Get value for a statistic interpreted as time
 */
inline double TimeStats::get_time(int id) {
    assert( id >= 0 && id < (int)value.size() );
    return ((double)(value[id])) / (double)CLOCKS_PER_SEC;
}


/**
 * Check and return id
 */
inline int TimeStats::get_id(const char* name) {
    assert( name_to_id[name].id != UNITIALIZED_STAT );
    return (name_to_id[name].id);
}


#endif /* STATS_HPP_ */
