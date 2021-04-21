#!/bin/bash

#SBATCH --job-name=train_classifier
#SBATCH --open-mode=append
#SBATCH --output=logs/train_cl_%j.out
#SBATCH --error=logs/train_cl_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x 

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/st3255/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"

#cd $HOME/test
#cd scratch/$USER/test

singularity exec --nv \
--bind /scratch \
--overlay /scratch/st3255/DL21SP/conda.sqsh:ro \
--overlay /tmp/student_dataset.sqsh:ro \
/scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dev
python train_classifier.py --checkpoint_dir checkpoints --encoder_checkpoint checkpoints/simclr_resnet18_512t01adam4_encoder
"
