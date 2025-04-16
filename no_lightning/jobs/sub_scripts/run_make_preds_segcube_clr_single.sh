#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 600
#SBATCH --gres=gpu:1
#SBATCH --exclude=compute-gpu-0-[0,1]
#SBATCH --error=/home/awilkins/contrastive-neutrino/no_lightning/jobs/logs/err/job.%x.%j.%A_%a.err
#SBATCH --output=/home/awilkins/contrastive-neutrino/no_lightning/jobs/logs/out/job.%x.%j.%A_%a.out

CONFIG_PATH=$1
CHKPT_PATH=$2
TEST_DIR_PATH=$3
FINETUNE_PICKLE_PATH=$4
XTALK=$5
PREDS_NAME=$6

echo "Job id ${SLURM_JOB_ID}"
echo "Job array task id ${SLURM_ARRAY_TASK_ID}"
echo "Node ${SLURMD_NODENAME}"
echo "GPU id ${CUDA_VISIBLE_DEVICES}"
echo "Config path ${CONFIG_PATH}"
echo "Chkpt path ${CHKPT_PATH}"
echo "Test dir path ${TEST_DIR_PATH}"
echo "Finetune pickle path ${FINETUNE_PICKLE_PATH}"
echo "Evaluation xtalk ${XTALK}"
echo "Outputted prediction name ${PREDS_NAME}"

cd /home/awilkins/contrastive-neutrino/no_lightning

apptainer exec --nv \
               --bind /share/ \
               /share/rcifdata/awilkins/images/rradev_minkowski_sandbox/ \
               python make_preds.py --batch_mode \
                                    --clr \
                                    --small_output \
                                    --xtalk $XTALK \
                                    --finetune_pickle $FINETUNE_PICKLE_PATH \
                                    $CONFIG_PATH \
                                    $CHKPT_PATH \
                                    $TEST_DIR_PATH \
                                    $PREDS_NAME
