#!/bin/bash
#SBATCH -p GPU
#SBATCH -N1
#SBATCH -c8
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=compute-gpu-0-[0,1,3,5]
#SBATCH --error=/home/awilkins/contrastive-neutrino/no_lightning/jobs/logs/err/job.%x.%j.err
#SBATCH --output=/home/awilkins/contrastive-neutrino/no_lightning/jobs/logs/out/job.%x.%j.out

CONFIG_FILE=$1
CHKPT_PATH=$2

NOMINAL_DATASET="/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube"
NOMINAL_XTALK=0.5

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES
echo $CONFIG_FILE

cd /home/awilkins/contrastive-neutrino/no_lightning

apptainer exec --nv \
               --bind /share/ \
               /share/rcifdata/awilkins/images/rradev_minkowski_sandbox/ \
               python train.py --print_iter 50 $CONFIG_FILE

apptainer exec --nv \
               --bind /share/ \
               /share/rcifdata/awilkins/images/rradev_minkowski_sandbox/ \
               python fine_tune_clr.py --pickle_model \
                                       --max_iter 300 \
                                       --no_augs \
                                       --nominal_xtalk $NOMINAL_XTALK \
                                       $CONFIG_FILE \
                                       $NOMINAL_DATASET \
                                       ${CHKPT_PATH}/net_best_epoch*

