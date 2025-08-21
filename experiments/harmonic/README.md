# Harmonic Series Numerical Analysis

This experiment analyzes the numerical accuracy and convergence properties of harmonic series computations using various floating-point error analysis tools.

## Reproduction Steps

1. **Build container**: Use `../../containers/Dockerfile-harmonic` to build the harmonic container
2. **Run experiment**: Execute `./launch_slurm.sh` (requires SLURM scheduler) or run the container directly:
   ```bash
   apptainer exec --env LD_LIBRARY_PATH=/usr/local/lib --pwd /build/harmonic \
     -B $PWD/results:/build/harmonic/.result ../../containers/fuzzy-pytorch-harmonic.sif ./run.sh
   ```
3. **Generate results**: Run `numerical-harmonic.ipynb` to analyze data from the `results/` directory

## Outputs

- `results/`: Raw numerical results from different tools and iteration counts
- `harmonic-numerics.csv`: Processed statistics (mean, std) for each tool/iteration combination
- `figures/`: Generated visualizations comparing numerical accuracy
- `numerical-harmonic.ipynb`: Analysis notebook generating plots from raw results