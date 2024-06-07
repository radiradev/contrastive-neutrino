#!/bin/bash
#SBATCH --image=rradev/minkowski:torch1.12_final
#SBATCH -A dune_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND="cores"
srun shifter python3 train.py --model sim_clr --batch_size 4096 --num_of_gpus 1 --dataset_type contrastive --gather_distributed False rradev/model-registry/resclr-throw-aug:latest
