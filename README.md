# Fuzzy PyTorch: Rapid Numerical Variability Evaluation for Deep Learning Models

This repository contains the experimental code and reproducibility artifacts for the paper "Fuzzy PyTorch: Rapid Numerical Variability Evaluation for Deep Learning Models".

## Overview

Fuzzy PyTorch provides tools and methodologies for evaluating numerical variability and floating-point precision effects in deep learning models. This repository contains three main experimental evaluations comparing various floating-point error analysis tools:

- **Deep Learning Benchmarks**: Performance evaluation on real-world ML models (MNIST, FastSurfer, WavLM)
- **Harmonic Series Analysis**: Numerical accuracy assessment using mathematical series computations
- **NAS Parallel Benchmarks**: Performance overhead analysis using standard HPC benchmarks

## Repository Structure

```
├── containers/           # Container definitions for reproducible experiments
│   ├── Dockerfile-NPB   # NPB benchmarks environment
│   ├── Dockerfile-harmonic  # Harmonic series environment  
│   ├── Dockerfile-tools # Deep learning environment
│   └── NPB/             # NPB source and build scripts
├── experiments/         # Experimental evaluations
│   ├── DL/              # Deep learning performance evaluation
│   ├── NPB/             # NAS Parallel Benchmarks analysis
│   └── harmonics/       # Harmonic series numerical analysis
└── README.md           # This file
```

## Experiments

### 1. Deep Learning Performance ([experiments/DL/](experiments/DL/))
Evaluates the runtime overhead of floating-point error analysis tools on deep learning workloads including MNIST classification, FastSurfer brain segmentation, and WavLM speech processing.

### 2. Harmonic Series Analysis ([experiments/harmonics/](experiments/harmonics/))
Assesses numerical accuracy and convergence properties of harmonic series computations across different floating-point precision analysis methods.

### 3. NAS Parallel Benchmarks ([experiments/NPB/](experiments/NPB/))
Measures performance overhead of various floating-point error analysis tools using the standard NAS Parallel Benchmarks suite (BT, CG, EP, FT, LU, MG, SP).

## Prerequisites

- **Apptainer/Singularity**: Container runtime for reproducible execution
- **SLURM** (optional): For automated batch job execution
- **Python 3.x**: For analysis notebooks
- **Jupyter**: For running analysis notebooks

## Quick Start

1. **Build containers**:
   ```bash
   cd containers
   ./build.sh
   ```

2. **Run experiments**: Navigate to each experiment directory and follow the instructions in their respective README files.

3. **Generate figures**: Execute the Jupyter notebooks in each experiment directory to reproduce the analysis and figures.

## Tools Evaluated

The experiments compare the following floating-point error analysis tools:

- **IEEE**: Standard IEEE 754 floating-point arithmetic
- **PRISM**: Precision analysis with stochastic rounding variants (SR, UD)
- **Verrou**: Monte Carlo Arithmetic with CESTAC and SR modes
- **CADNA/CESTAC**: Control of Accuracy and Debugging for Numerical Applications
- **Verificarlo**: Monte Carlo Arithmetic with Random Rounding (MCA RR)
- **FM SR**: Fast Math Stochastic Rounding

## Citation

If you use this work, please cite:
```
Fuzzy PyTorch: Rapid Numerical Variability Evaluation for Deep Learning Models
```

## License

See individual component licenses in their respective directories.