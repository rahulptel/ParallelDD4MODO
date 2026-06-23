#!/bin/bash
# A single case computation, for the case No. $2 in the table $1.

TABLE1=$1
i1=$2

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FARM_NAME="$(basename "$SCRIPT_DIR")"
DEFAULT_ROOT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$DEFAULT_ROOT_DIR}"
RUNS_DIR="$ROOT_DIR/outputs/$FARM_NAME"
STATUS_DIR="$SCRIPT_DIR/STATUSES"

# Total number of cases:
## If the env. variable N_cases is defined, using it, otherwise computing the number of lines in the table:
if test -z $N_cases
  then
  N_cases=`cat "$TABLE1" | wc -l`
  fi

# Exiting if $i1 goes beyond $N_cases (can happen in bundled farms):
if test $i1 -lt 1 -o $i1 -gt $N_cases  
  then
  exit
  fi
  
# Extracing the $i1-th line from file $TABLE1:
LINE=`sed -n ${i1}p $TABLE1`
# Case id (from the original cases table):
ID=`echo "$LINE" | cut -d" " -f1`
# The rest of the line:
COMM=`echo "$LINE" | cut -d" " -f2-`

METAJOB_ID=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# ++++++++++++++++++++++  This part can be customized:  ++++++++++++++++++++++++
#  Here:
#  $ID contains the case id from the original table (can be used to provide a unique seed to the code etc)
#  $COMM is the line corresponding to the case $ID in the original table, without the ID field
#  $METAJOB_ID is the jobid for the current meta-job (convenient for creating per-job files)

# If you do not want each case to be computed inside a separate subdirectory, comment out the following two lines
# and also comment out the line that changes back to the farm directory below.
mkdir -p "$RUNS_DIR/RUN$ID"
cd "$RUNS_DIR/RUN$ID"

echo "Case $ID:"

# Executing the command (a line from table.dat)
# It's allowed to use more than one shell command (separated by semi-columns) on a single line
CASE_MEM_LIMIT_BYTES=$((16 * 1024 * 1024 * 1024))
CASE_TIME_LIMIT=1h
STATUS_MEM_LIMIT=90
STATUS_TIME_LIMIT=91
STATUS_LIMITER_UNAVAILABLE=88
STATUS_TIMEOUT_UNAVAILABLE=87
CASE_STDERR_FILE="case.stderr"
CASE_STDOUT_FILE="case.stdout"

if ! command -v prlimit >/dev/null 2>&1
then
  echo "Case $ID: error - prlimit command not found; refusing to run without per-case memory limit."
  STATUS=$STATUS_LIMITER_UNAVAILABLE
elif ! command -v timeout >/dev/null 2>&1
then
  echo "Case $ID: error - timeout command not found; refusing to run without per-case time limit."
  STATUS=$STATUS_TIMEOUT_UNAVAILABLE
else
  : > "$CASE_STDERR_FILE"
  : > "$CASE_STDOUT_FILE"
  setsid -w timeout --signal=TERM --kill-after=5s $CASE_TIME_LIMIT \
    prlimit --as=$CASE_MEM_LIMIT_BYTES -- bash -lc "$COMM" > "$CASE_STDOUT_FILE" 2> "$CASE_STDERR_FILE" &
  SESSION_PID=$!
  wait $SESSION_PID
  STATUS=$?
  kill -9 -$SESSION_PID 2>/dev/null

  if test $STATUS -eq 124
  then
    echo "Case $ID: time-limit failure detected (limit=$CASE_TIME_LIMIT) -> STATUS=$STATUS_TIME_LIMIT"
    STATUS=$STATUS_TIME_LIMIT
  elif test $STATUS -ne 0
  then
    IS_GPU_CASE=0
    if echo "$COMM" | grep -Eq '(^|[[:space:]])--backend[[:space:]]+gpu([[:space:]]|$)|(^|[[:space:]])gpu([[:space:]]|$)'
    then
      IS_GPU_CASE=1
    fi

    CPU_OOM_PATTERN='std::bad_alloc|cannot allocate memory|out of memory|oom'
    GPU_OOM_PATTERN='cudaErrorMemoryAllocation|CUDA_ERROR_OUT_OF_MEMORY|CUBLAS_STATUS_ALLOC_FAILED|thrust::system::system_error|hipErrorOutOfMemory|failed to allocate device memory|CUDA out of memory|out of memory'

    OOM_MATCHED=0
    if grep -Eiq "$CPU_OOM_PATTERN" "$CASE_STDERR_FILE" "$CASE_STDOUT_FILE"
    then
      OOM_MATCHED=1
    elif test $IS_GPU_CASE -eq 1 && grep -Eiq "$GPU_OOM_PATTERN" "$CASE_STDERR_FILE" "$CASE_STDOUT_FILE"
    then
      OOM_MATCHED=1
    fi

    if test $STATUS -eq 134 -o $STATUS -eq 137 -o $STATUS -eq 139 -o $OOM_MATCHED -eq 1
    then
      echo "Case $ID: memory-limit failure detected (exit=$STATUS) -> STATUS=$STATUS_MEM_LIMIT"
      STATUS=$STATUS_MEM_LIMIT
    fi
  fi
fi

# Comment out this line if not creating a separate subdirectory for each case:
cd "$SCRIPT_DIR"

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Saving all current metajob statuses to a single file with a unique name:
mkdir -p "$STATUS_DIR"
echo $ID $STATUS >> "$STATUS_DIR/status.$METAJOB_ID"
