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

# Building Docker images for MNIST test or as a base
docker build . -f Dockerfile-pytorch-sr -t big-data-lab-team/fuzzy-pytorch:sr

# Convert to apptainer
apptainer build fuzzy-pytorch-sr.sif docker-daemon:big-data-lab-team/fuzzy-pytorch:sr

# Building Docker images for Fastsurfer test 
docker build . -f Dockerfile-fastsurfer -t big-data-lab-team/fuzzy-fastsurfer:sr

# Convert to apptainer
apptainer build fuzzy-fastsurfer-sr.sif docker-daemon:big-data-lab-team/fuzzy-fastsurfer:sr

# Building Docker images for WavLM test 
docker build . -f Dockerfile-wavlm -t big-data-lab-team/fuzzy-wavlm:sr

# Convert to apptainer
apptainer build fuzzy-wavlm-sr.sif docker-daemon:big-data-lab-team/fuzzy-wavlm:sr