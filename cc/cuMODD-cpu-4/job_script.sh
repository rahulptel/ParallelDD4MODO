#!/bin/bash

# Here you should provide the sbatch arguments to be used in all jobs in this farm
# At the very least, it has to contain the runtime switch (either -t or --time):
#SBATCH -t 0-03:00
#SBATCH --mem=17G
#SBATCH --cpus-per-task=4
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# If WHOLE_NODE=1 in config.h file, the following sbatch arguments will be automatically added:
# --nodes=1 --cpus-per-task=$NWHOLE --exclusive , where $NWHOLE is also defined in config.h


# Don't change the lines below
#=====================================================================

module load meta-farm
module load boost
module load cuda

if (($WHOLE_NODE==1))
 then
 for ((i=0; i<$SLURM_CPUS_PER_TASK; i++))
  do
  task.run &
  done
 wait

 else
 task.run

 fi
