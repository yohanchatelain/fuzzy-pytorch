# Deep Learning Performance Evaluation

This experiment evaluates the performance overhead of various floating-point error analysis tools on deep learning benchmarks.

## Reproduction Steps

1. **Prerequisites**: Container built with `../../containers/Dockerfile-tools`
2. **Run experiments**: Execute the container with the appropriate SLURM configuration
3. **Generate results**: Run `perf-DL.ipynb` to analyze performance data from `perf-DL-stats.csv`

## Outputs

- `perf-DL-stats.csv`: Performance statistics for MNIST, FastSurfer, and WavLM benchmarks
- `figures/figure-3.png`: Runtime comparison visualization
- `perf-DL.ipynb`: Analysis notebook generating the performance plots