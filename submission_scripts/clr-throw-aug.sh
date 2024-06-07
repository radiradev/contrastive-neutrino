#!/bin/bash
#SBATCH --image=rradev/minkowski:torch1.12_final
#SBATCH -A m4709
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND="cores"
srun shifter python3 train.py --config-name contrastive_augmentations_throws