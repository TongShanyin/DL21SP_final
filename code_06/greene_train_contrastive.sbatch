#!/bin/bash

#SBATCH --job-name=train_simclr
#SBATCH --open-mode=append
#SBATCH --output=logs/train_simclr.out
#SBATCH --error=logs/train_simclr.err
#SBATCH --time=80:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x 

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/$USER/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"

singularity exec --nv \
--bind /scratch \
--overlay /scratch/$USER/DL21SP/conda.sqsh:ro \
--overlay /tmp/student_dataset.sqsh:ro \
/scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dev
python train_contrastive.py --checkpoint-dir checkpoints 
"
