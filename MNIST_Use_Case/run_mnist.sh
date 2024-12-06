#!/bin/bash
#SBATCH --job-name=fuzzy_mnist_test_singlethread
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --array=6-10
#SBATCH --output=slurm/%x_%a.out

export OMP_NUM_THREADS=1
export NUMPEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TASK_ID=$SLURM_ARRAY_TASK_ID

# NAN Inference
#time apptainer exec --env THRESHOLD=0.5 --env EPSILON=1e-7 -B ../mnist/:/mnist mnist.sif python3 /mnist/nan_mnist_test.py --load-model --model-path /mnist/mnist_cnn_pool.pt 


# Normal Inference
#time apptainer exec -B ../mnist:/mnist mnist.sif python3 /mnist/mnist_test.py --load-model --model-path /mnist/mnist_cnn.pt

# Fuzzy PyTorch Inference -- takes approx 3 mins
time apptainer exec -B ../mnist:/mnist fuzzy_pytorch.sif python3 /mnist/mnist_test.py --load-model --model-path /mnist/mnist_cnn.pt

#Verrou Inference
#time apptainer exec -B ../mnist:/mnist verrou_mnist.sif valgrind --tool=verrou --rounding-mode=random --mca-mode=rr -s --check-nan=no python3 /mnist/mnist_test.py --load-model --model-path /mnist/mnist_cnn.pt

#Verrou Training
#time apptainer exec -B ../mnist/:/mnist verrou_mnist.sif valgrind --tool=verrou --rounding-mode=random --mca-mode=rr -s --check-nan=no python3 /mnist/mnist_train.py --save-model
