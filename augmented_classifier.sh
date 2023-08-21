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
srun shifter python3 -m single_particle_classifier.classifier_train --batch_size 256 --num_of_gpus 4 --dataset_type single_particle_augmented --checkpoint /global/homes/r/rradev/contrastive-neutrino/artifacts/augmented-classifier/model.ckpt