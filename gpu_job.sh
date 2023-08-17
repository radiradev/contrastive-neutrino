#!/bin/bash
#SBATCH --image=rradev/minkowski:torch1.12_final
#SBATCH -A dune_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND="cores"
srun shifter python3 train.py --batch_size 128
