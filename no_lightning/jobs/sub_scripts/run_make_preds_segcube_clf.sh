#!/bin/bash
#SBATCH -p GPU
#SBATCH -N1
#SBATCH -c8
#SBATCH --gres=gpu:1
#SBATCH --exclude=compute-gpu-0-[0,1,3]
#SBATCH --error=/home/awilkins/contrastive-neutrino/no_lightning/jobs/logs/err/job.%x.%j.err
#SBATCH --output=/home/awilkins/contrastive-neutrino/no_lightning/jobs/logs/out/job.%x.%j.out

CONFIG_PATH=$1
CHKPT_PATH=$2
TEST_DIR_PATH=$3

echo $0
echo "node: ${SLURMD_NODENAME}"
echo "gpu id: ${CUDA_VISIBLE_DEVICES}"
echo "config_path: ${CONFIG_path}"
echo "chkpt_path: ${CHKPT_PATH}"
echo "test_dir_path: ${TEST_DIR_PATH}"

cd /home/awilkins/contrastive-neutrino/no_lightning

for xtalk_num in {0,05,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95}
do
  echo $xtalk_num

  xtalk=0.${xtalk_num}
  preds_name=preds_xtalk${xtalk_num}.yml

  apptainer exec --nv \
                 --bind /share/ \
                 /share/rcifdata/awilkins/images/rradev_minkowski_sandbox/ \
                 python make_preds.py --batch_mode \
                                      --classifier \
                                      --xtalk $xtalk \
                                      $CONFIG_PATH \
                                      $CHKPT_PATH \
                                      $TEST_DIR_PATH \
                                      $preds_name
done

echo 100

xtalk=1.0
preds_name=preds_xtalk100.yml

apptainer exec --nv \
               --bind /share/ \
               /share/rcifdata/awilkins/images/rradev_minkowski_sandbox/ \
               python make_preds.py --batch_mode \
                                    --classifier \
                                    --xtalk $xtalk \
                                    $CONFIG_PATH \
                                    $CHKPT_PATH \
                                    $TEST_DIR_PATH \
                                    $preds_name
