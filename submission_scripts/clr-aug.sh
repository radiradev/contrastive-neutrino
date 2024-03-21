#!/bin/bash
#SBATCH --image=rradev/minkowski:torch1.12_final
#SBATCH -A dune_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND="cores"
srun shifter python3 train.py --model sim_clr --batch_size 128 --num_of_gpus 4 --dataset_type augmentations --gather_distributed False \
 --wandb_checkpoint rradev/model-registry/contrastive-augmentations:latest
