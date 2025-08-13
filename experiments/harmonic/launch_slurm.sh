#!/bin/bash

CONTAINER_IMAGE=../../containers/fuzzy-pytorch-harmonic.sif

mkdir -p results

sbatch --time 00:30:00 --mem 1GB --ntasks 64 --wrap "apptainer exec --env LD_LIBRARY_PATH=/usr/local/lib --pwd /build/harmonic -B $PWD/results:/build/harmonic/.result $CONTAINER_IMAGE ./run.sh"
