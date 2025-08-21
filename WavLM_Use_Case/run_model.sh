#!/bin/bash
#SBATCH --mem-per-cpu=3G
#SBATCH --ntasks=1
#SBATCH --time=3:0:0
#SBATCH --array=30,31,32,33,34,36,38,39,40,42,43,44,45,46,47,48,50,51,53,54,55,57,58,60,61,62,64,65,66,67,68
#SBATCH --account=def-glatard
#SBATCH --output=static_output/%x_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inesgp99@gmail.com

subject_vals=(
5395675
5540620
5473776
5713377
6044197
6023682
5593380
6006405
5801628
5965463
5890374
5809299
5762345
5824259
5439390
5492841
5905392
5822338
5561646
5933409
5610075
5888115
5945096
5649136
5670303
5833952
5935026
6028564
5441506
5983944
5747486
5946889
5566601
5976949
5403980
5714015
5466129
5408212
5869483
5402792
5453241
5872967
5531834
5890300
5396986
5569833
5413312
5763007
5887447
5757752
5715968
5404958
5788655
5893493
5837015
5534714
5781428
5772616
5934501
5962426
5962276
5631826
5931797
5781515
5884447
5568881
5405698
5535312
)
echo "${subject_vals[(${SLURM_ARRAY_TASK_ID} - 1)]}"

module load apptainer/1.2

time apptainer exec --env FILEPATH='/workdir' --env DATAPATH='/data_dir' \
--env SUBJECT_ID=${subject_vals[(${SLURM_ARRAY_TASK_ID} - 1)]} \
--env OUTPUT_PATH='/workdir/static_embeddings' --env TOOL='fuzzy' --env MODE='ud' \
-B /scratch/hibaa/test_set/:/data_dir \
-B /scratch/ine5/fuzzy-pytorch/WavLM_Use_Case/:/workdir \
-B core.py:/usr/local/lib/python3.8/dist-packages/speechbrain/core.py \
/scratch/ine5/fuzzy_wavlm_ud_static.sif python3 /workdir/inference_only.py /workdir/train.yaml --data_folder=/workdir/test_matched 

#python3 load.py

