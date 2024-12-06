#!/bin/bash
#SBATCH --job-name=ieee_fastsurfer
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks-per-node=5
#SBATCH --nodes=1
#SBATCH --time=UNLIMITED
#SBATCH --array=1-5
#SBATCH --output=slurm/%x_%a.out

export OMP_NUM_THREADS=1
export NUMPEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

module load apptainer/1.2

parallel "{} > slurm/ieee_fastsurfer_{#}_${SLURM_ARRAY_TASK_ID}.log 2>&1" :::: ieee_subjects.txt


# time docker run -it --rm -v /home/inesgp/nanpool_results/data/:/data -v /home/inesgp/verrou_fastsurfer/sub-0025531:/output -v /home/inesgp/verrou_fastsurfer/:/fs_license inesgp/fuzzy_fastsurfer:v0.0 /bin/bash -c "/fastsurfer/run_fastsurfer.sh --t1 /data/0025531_norm.mgz --sid "0025531_${SLURM_ARRAY_TASK_ID}" --sd /output --fs_license /fs_license/license.txt --seg_only --parallel --device cpu"



