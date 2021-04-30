#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=n1s8-t4-1
#SBATCH --account=dl06
#SBATCH --time=05:00:00
#SBATCH --output=logs/submit/finetune_extra.out
#SBATCH --error=logs/submit/finetune_extra.err
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
python finetune.py --checkpoint_dir checkpoints/submit/finetune_extra --model_checkpoint checkpoints/submit/classifier_extra/extra_classifier_epoch67.pth
"
