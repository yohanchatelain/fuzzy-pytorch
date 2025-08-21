#!/bin/bash

# Loop through each MCA iteration and submit a job
for i in 5; do
     export ITER=$i
     # sbatch --job-name=ieee_torch_$i run_verrou.sh
     # sbatch --job-name=verrou_torch_ud_$i run_verrou2.sh
     # sbatch --job-name=verrou_torch_sr_$i run_verrou.sh
     sbatch --job-name=fuzzy_torch_ud_$i run_model.sh 
done
