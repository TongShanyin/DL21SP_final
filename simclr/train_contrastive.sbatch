#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=n1s8-t4-1
#SBATCH --account=dl06
#SBATCH --time=20:00:00
#SBATCH --mem=20GB
#SBATCH --output=logs/train_simclr_%j.out
#SBATCH --error=logs/train_simclr_%j.err
#SBATCH --exclusive
#SBATCH --no-requeue 

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x 

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"

#cd $HOME/test
cd /scratch/$USER/DL21SP_final/autoencoder

singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL21SP/conda.sqsh:ro \
--overlay /tmp/student_dataset.sqsh:ro \
/share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dev
python train_contrastive.py --checkpoint-dir checkpoints --checkpoint_net checkpoints/simclr_resnet18_start50_lr4_ep_10
"
#--checkpoint_net checkpoints/simclr_lr3_ep_25
#"
