#!/bin/bash

CONTAINER_IMAGE=../../containers/fuzzy-pytorch-npb.sif


# Create results directory if it doesn't exist
mkdir -p "${PWD}/run-SER"

# Loop through all combinations
for bench in bt cg ep ft is lu mg sp; do
  for class in S A; do
    echo "Running benchmark: $bench, class: $class"
    sbatch --mem 0 --ntasks 64 --output ${bench}_${class}.out --error ${bench}_${class}.err --wrap="apptainer exec --env LD_LIBRARY_PATH=/usr/local/lib --pwd /build/NPB -B $PWD/run-SER:/build/NPB/results ${CONTAINER_IMAGE} /build/NPB/run.sh $bench $class"
  done
done
