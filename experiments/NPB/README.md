# NAS Parallel Benchmarks (NPB) Performance Analysis

This experiment evaluates the performance overhead of floating-point error analysis tools using the NAS Parallel Benchmarks suite.

## Reproduction Steps

1. **Build container**: Use `../../containers/Dockerfile-NPB` to build the NPB container
2. **Run serial experiments**: Execute `./launch_slurm.sh` for serial benchmarks (requires SLURM scheduler)
3. **Run OpenMP experiments**: Execute `./launch_slurm_omp.sh` for parallel benchmarks
4. **Alternative manual execution**:
   ```bash
   # Create results directory
   mkdir -p run-SER
   # Run specific benchmark (example: BT class S)
   apptainer exec --env LD_LIBRARY_PATH=/usr/local/lib --pwd /build/NPB \
     -B $PWD/run-SER:/build/NPB/results ../../containers/fuzzy-pytorch-npb.sif \
     /build/NPB/run.sh bt S
   ```
5. **Generate analysis**: Run `perf-NPB-SER.ipynb` to analyze performance data

## Benchmarks

- **BT**: Block Tridiagonal solver
- **CG**: Conjugate Gradient
- **EP**: Embarrassingly Parallel
- **FT**: Fast Fourier Transform
- **IS**: Integer Sort (not included in current analysis)
- **LU**: LU decomposition
- **MG**: Multigrid
- **SP**: Scalar Pentadiagonal solver

**Classes**: S (small), A (standard)

## Outputs

- `run-SER/`: Raw benchmark results for each tool/benchmark/class combination
- `perf-NPB-SER-stats.csv`: Processed performance statistics and slowdown analysis
- `figures/figure2.png`: Performance comparison visualization
- `perf-NPB-SER.ipynb`: Analysis notebook generating performance plots