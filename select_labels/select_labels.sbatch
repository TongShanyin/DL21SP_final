#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=n1s8-t4-1
#SBATCH --account=dl06
#SBATCH --time=00:20:00
#SBATCH --output=outputs/select_%j.out
#SBATCH --error=outputs/select_%j.err
#SBATCH --exclusive
#SBATCH --requeue 

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x 

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"

#cd $HOME/test
#cd scratch/$USER/test

singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL21SP/conda.sqsh:ro \
--overlay /tmp/student_dataset.sqsh:ro \
/share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dev
python select_labels.py --model_checkpoint checkpoints/classifier_simclr_resnet18_512t01adam4_encoder_epoch20.pth
"
#--encoder_checkpoint checkpoints/simclr_resnet18_start50_lr4_encoder_ep_10
#" # 5%
#--encoder_checkpoint checkpoints/simclr_lr4_encoder_ep_30
#"
#--encoder_checkpoint checkpoints/simclr_resnet18_encoder_ep_50
#"
