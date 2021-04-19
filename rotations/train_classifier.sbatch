#!/bin/bash

#SBATCH --job-name=dummy_train_classifier
#SBATCH --open-mode=append
#SBATCH --output=c_logs/train_dummy_%j.out
#SBATCH --error=c_logs/train_dummy_%j.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x 

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/ccp5804/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"

#cd $HOME/test
cd scratch/$USER/test

singularity exec --nv \
--bind /scratch \
--overlay /scratch/ccp5804/DL21SP/conda.sqsh:ro \
--overlay /tmp/student_dataset.sqsh:ro \
/scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dev
python train_classifier.py --checkpoint_dir checkpoints --alexnet_checkpoint checkpoints/rotations_aug/rotations_ep_80
"
