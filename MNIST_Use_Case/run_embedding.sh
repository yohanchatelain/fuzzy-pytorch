#!/bin/bash
#SBATCH --job-name=verrou_embed
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --array=1-10
#SBATCH --output=slurm/%x_%a.out

export OMP_NUM_THREADS=1
export NUMPEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TASK_ID=$SLURM_ARRAY_TASK_ID

# Normal Inference
#time apptainer exec -B ../mnist:/mnist mnist.sif python3 /mnist/mnist_test_embeddings.py --load-model --model-path /mnist/mnist_cnn.pt


# Fuzzy PyTorch Inference -- takes approx 3 mins
#time apptainer exec -B ../mnist:/mnist fuzzy_pytorch.sif python3 /mnist/mnist_test_embeddings.py --load-model --model-path /mnist/mnist_cnn.pt

#Verrou Inference
time apptainer exec -B ../mnist:/mnist verrou_mnist.sif valgrind --tool=verrou --rounding-mode=random --mca-mode=rr -s --check-nan=no python3 /mnist/mnist_test_embeddings.py --load-model --model-path /mnist/mnist_cnn.pt
