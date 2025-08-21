#!/bin/bash

CONTAINER_IMAGE=big-data-lab-team/fuzzy-pytorch:paper-experiments-npb

# Create results directory if it doesn't exist
mkdir -p "${PWD}/run-OMP"

# Loop through all combinations
for class in S A; do
  for tool in ieee prism_sr_static prism_ud_static prism_sr_dynamic prism_ud_dynamic cadna verificarlo verrou_cestac verrou_sr sr; do
    for bench in bt cg ep ft is lu mg sp; do
      for threads in 2 4 8 16 32; do
        echo "Running tool: $tool, benchmark: $bench, class: $class, threads: $threads"
        sbatch --mem 0 --ntasks 64 --output ${tool}_${bench}_${class}_${threads}.out --error ${tool}_${bench}_${class}_${threads}.err --wrap="apptainer exec --env LD_LIBRARY_PATH=/usr/local/lib --pwd /build/NPB -B $PWD/run-OMP:/build/NPB/results ${CONTAINER_IMAGE} /build/NPB/run_omp.sh $tool $bench $class $threads"
      done
    done
  done
done
