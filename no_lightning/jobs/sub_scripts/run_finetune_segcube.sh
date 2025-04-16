#!/bin/bash
#SBATCH -p GPU
#SBATCH -N1
#SBATCH -c8
#SBATCH --gres=gpu:1
#SBATCH --error=/home/awilkins/contrastive-neutrino/no_lightning/jobs/logs/err/job.%x.%j.err
#SBATCH --output=/home/awilkins/contrastive-neutrino/no_lightning/jobs/logs/out/job.%x.%j.out

CONFIG_FILE=$1
CHKPT_PATH=$2
DATASET_PATH=$3
XTALK=$4
PICKLE_NAME=$5

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES
echo $CONFIG_FILE

cd /home/awilkins/contrastive-neutrino/no_lightning

apptainer exec --nv \
               --bind /share/ \
               /share/rcifdata/awilkins/images/rradev_minkowski_sandbox/ \
               python fine_tune_clr.py --pickle_model \
                                       --pickle_name $PICKLE_NAME \
                                       --max_iter 300 \
                                       --nominal_xtalk $XTALK \
                                       --no_augs \
                                       $CONFIG_FILE \
                                       $DATASET_PATH \
                                       ${CHKPT_PATH}/net_best_epoch*

