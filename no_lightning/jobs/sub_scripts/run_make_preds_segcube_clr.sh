#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 600
#SBATCH --gres=gpu:1
#SBATCH --array=0-20
#SBATCH --exclude=compute-gpu-0-[0,1,3]
#SBATCH --error=/home/awilkins/contrastive-neutrino/no_lightning/jobs/logs/err/job.%x.%j.err
#SBATCH --output=/home/awilkins/contrastive-neutrino/no_lightning/jobs/logs/out/job.%x.%j.out

CONFIG_PATH=$1
CHKPT_PATH=$2
TEST_DIR_PATH=$3
FINETUNE_PICKLE_PATH=$4

echo "Job id ${SLURM_JOB_ID}"
echo "Job array task id ${SLURM_ARRAY_TASK_ID}"
echo "Node ${SLURMD_NODENAME}"
echo "GPU id ${CUDA_VISIBLE_DEVICES}"
echo "Config path ${CONFIG_PATH}"
echo "Chkpt path ${CHKPT_PATH}"
echo "Test dir path ${TEST_DIR_PATH}"
echo "Finetune pickle path ${FINETUNE_PICKLE_PATH}"

xtalks=("0" "0.05" "0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4" "0.45" "0.5" "0.55" "0.6" "0.65" "0.7" "0.75" "0.8" "0.85" "0.9" "0.95" "1.0")
xtalk_suffixes=(0 05 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)
xtalk=${xtalks[${SLURM_ARRAY_TASK_ID}]}
xtalk_suffix=${xtalk_suffixes[${SLURM_ARRAY_TASK_ID}]}
preds_name=preds_xtalk${xtalk_suffix}

echo "Xtalk $xtalk"
echo "Preds out name $preds_name"

cd /home/awilkins/contrastive-neutrino/no_lightning

apptainer exec --nv \
               --bind /share/ \
               /share/rcifdata/awilkins/images/rradev_minkowski_sandbox/ \
               python make_preds.py --batch_mode \
                                    --clr \
                                    --small_output \
                                    --xtalk $xtalk \
                                    --finetune_pickle $FINETUNE_PICKLE_PATH \
                                    $CONFIG_PATH \
                                    $CHKPT_PATH \
                                    $TEST_DIR_PATH \
                                    $preds_name
