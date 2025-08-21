#!/bin/bash

set -e
set -o pipefail

# Building Docker images for tools used in experiments
docker build . -f Dockerfile-tools -t big-data-lab-team/fuzzy-pytorch:paper-experiments-tools

# Building Docker images for Harmonic test
docker build . -f Dockerfile-harmonic -t big-data-lab-team/fuzzy-pytorch:paper-experiments-harmonic

# Convert to apptainer
apptainer build fuzzy-pytorch-harmonic.sif docker-daemon:big-data-lab-team/fuzzy-pytorch:paper-experiments-harmonic

# Building Docker images for NPB test
docker build . -f Dockerfile-NPB -t big-data-lab-team/fuzzy-pytorch:paper-experiments-npb

# Convert to apptainer
apptainer build fuzzy-pytorch-npb.sif docker-daemon:big-data-lab-team/fuzzy-pytorch:paper-experiments-npb
