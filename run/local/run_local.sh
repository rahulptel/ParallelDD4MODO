#!/bin/bash
# Local runner for a given table configuration with optional filters.
# Usage: ./run_local.sh <farm_name_or_path_to_dat> [filters]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TABLE_ARG=""
PROBLEM_FILTER=""
OBJECTIVES_FILTER=""
VARIABLES_FILTER=""
SEED_FILTER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--problem)
            if [ -z "${2:-}" ]; then
                echo "Error: --problem requires a value."
                exit 1
            fi
            PROBLEM_FILTER="$2"
            shift 2
            ;;
        -k|--objectives)
            if [ -z "${2:-}" ]; then
                echo "Error: --objectives requires a value."
                exit 1
            fi
            OBJECTIVES_FILTER="$2"
            shift 2
            ;;
        -n|--variables)
            if [ -z "${2:-}" ]; then
                echo "Error: --variables requires a value."
                exit 1
            fi
            VARIABLES_FILTER="$2"
            shift 2
            ;;
        -s|--seed)
            if [ -z "${2:-}" ]; then
                echo "Error: --seed requires a value."
                exit 1
            fi
            SEED_FILTER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 <farm_name_or_path_to_dat> [options]"
            echo "Options:"
            echo "  -p, --problem <val>     Filter by problem type (1=knapsack, 2=set packing, 3=tsp)"
            echo "  -k, --objectives <val>  Filter by number of objectives (3-7)"
            echo "  -n, --variables <val>   Filter by number of variables/cities"
            echo "  -s, --seed <val>        Filter by seed/instance number"
            exit 0
            ;;
        *)
            if [ -z "$TABLE_ARG" ]; then
                TABLE_ARG="$1"
                shift
            else
                echo "Error: Unknown argument: $1"
                exit 1
            fi
            ;;
    esac
done

if [ -z "$TABLE_ARG" ]; then
    echo "Error: Missing table configuration or farm name."
    echo "Usage: $0 <farm_name_or_path_to_dat> [options]"
    exit 1
fi

# Resolve the table path
if [ -f "$TABLE_ARG" ]; then
    TABLE_PATH="$(realpath "$TABLE_ARG")"
    FARM_NAME="$(basename "$TABLE_PATH" .dat)"
elif [ -f "$SCRIPT_DIR/$TABLE_ARG.dat" ]; then
    TABLE_PATH="$SCRIPT_DIR/$TABLE_ARG.dat"
    FARM_NAME="$TABLE_ARG"
else
    echo "Error: Cannot find table configuration for '$TABLE_ARG'."
    echo "Expected a file path or a farm name having a corresponding .dat file in $SCRIPT_DIR."
    exit 1
fi

OUTPUTS_DIR="$PROJECT_ROOT/outputs/local/$FARM_NAME"

echo "Running local execution for: $FARM_NAME"
[ -n "$PROBLEM_FILTER" ] && echo "  Filter: problem = $PROBLEM_FILTER"
[ -n "$OBJECTIVES_FILTER" ] && echo "  Filter: k = $OBJECTIVES_FILTER"
[ -n "$VARIABLES_FILTER" ] && echo "  Filter: n = $VARIABLES_FILTER"
[ -n "$SEED_FILTER" ] && echo "  Filter: seed = $SEED_FILTER"
echo "Reading cases from $TABLE_PATH..."

LINE_NUM=0
RUN_COUNT=0

while IFS= read -r cmd || [ -n "$cmd" ]; do
    # Skip empty lines
    if [ -z "$cmd" ]; then
        continue
    fi
    LINE_NUM=$((LINE_NUM + 1))

    # Parse metadata from cmd line
    # Format: BINARY INST_PATH PROBLEM_TYPE ...
    read -r -a tokens <<< "$cmd"
    binary="${tokens[0]}"
    inst_path="${tokens[1]}"
    problem_type="${tokens[2]}"
    filename="$(basename "$inst_path")"

    # Extract objectives
    nobjs=""
    if [[ "$binary" =~ multiobj_nobjs([3-7]) ]]; then
        nobjs="${BASH_REMATCH[1]}"
    fi

    # Extract variables
    nvars=""
    if [[ "$filename" =~ _n-([0-9]+)_ ]]; then
        nvars="${BASH_REMATCH[1]}"
    elif [[ "$filename" =~ knapsack-([0-9]+)- ]]; then
        nvars="${BASH_REMATCH[1]}"
    elif [[ "$filename" =~ bp-([0-9]+)- ]]; then
        nvars="${BASH_REMATCH[1]}"
    elif [[ "$filename" =~ ncities([0-9]+) ]]; then
        nvars="${BASH_REMATCH[1]}"
    fi

    # Extract seed
    seed=""
    if [[ "$filename" =~ _ins-([0-9]+) ]]; then
        seed="${BASH_REMATCH[1]}"
    elif [[ "$filename" =~ -([0-9]+)\.dat$ ]]; then
        seed="${BASH_REMATCH[1]}"
    elif [[ "$filename" =~ seed([0-9]+) ]]; then
        seed="${BASH_REMATCH[1]}"
    fi

    # Apply filters
    if [ -n "$PROBLEM_FILTER" ]; then
        if [ "$PROBLEM_FILTER" = "1" ] || [ "${PROBLEM_FILTER,,}" = "knapsack" ] || [ "${PROBLEM_FILTER,,}" = "kp" ]; then
            if [ "$problem_type" != "1" ]; then continue; fi
        elif [ "$PROBLEM_FILTER" = "2" ] || [ "${PROBLEM_FILTER,,}" = "setpacking" ] || [ "${PROBLEM_FILTER,,}" = "set_packing" ] || [ "${PROBLEM_FILTER,,}" = "bp" ] || [ "${PROBLEM_FILTER,,}" = "binproblem" ]; then
            if [ "$problem_type" != "2" ]; then continue; fi
        elif [ "$PROBLEM_FILTER" = "3" ] || [ "${PROBLEM_FILTER,,}" = "tsp" ]; then
            if [ "$problem_type" != "3" ]; then continue; fi
        else
            if [ "$PROBLEM_FILTER" != "$problem_type" ]; then continue; fi
        fi
    fi

    if [ -n "$OBJECTIVES_FILTER" ] && [ "$OBJECTIVES_FILTER" != "$nobjs" ]; then
        continue
    fi

    if [ -n "$VARIABLES_FILTER" ] && [ "$VARIABLES_FILTER" != "$nvars" ]; then
        continue
    fi

    if [ -n "$SEED_FILTER" ] && [ "$SEED_FILTER" != "$seed" ]; then
        continue
    fi

    RUN_COUNT=$((RUN_COUNT + 1))
    RUN_DIR="$OUTPUTS_DIR/RUN$LINE_NUM"
    mkdir -p "$RUN_DIR"

    echo "Running Case $LINE_NUM (matched case $RUN_COUNT)..."

    (
        cd "$RUN_DIR"
        if command -v timeout >/dev/null 2>&1; then
            timeout 1h bash -c "$cmd" > case.stdout 2> case.stderr
        else
            bash -c "$cmd" > case.stdout 2> case.stderr
        fi
    )
done < "$TABLE_PATH"

if [ "$RUN_COUNT" -eq 0 ]; then
    echo "Warning: No cases matched the specified filters."
else
    echo "Completed $RUN_COUNT matched cases for $FARM_NAME. Outputs saved in $OUTPUTS_DIR"
fi
