#!/bin/bash
#SBATCH --image=rradev/minkowski:torch1.12_final
#SBATCH -A dune_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 10:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
srun shifter python3 train.py --batch_size 128 --num_of_gpus 1 --dataset_type contrastive