#!/bin/bash
#SBATCH --job-name=verroufs_runtime
#SBATCH --mem-per-cpu=5G
#SBATCH --ntasks-per-node=5
#SBATCH --nodes=1
#SBATCH --time=UNLIMITED
#SBATCH --array=1-5
#SBATCH --account=rrg-glatard
#SBATCH --output=%x_%a.out

parallel "{} > runtime/{#}_${SLURM_ARRAY_TASK_ID}.log 2>&1" :::: allsub_fast.txt


