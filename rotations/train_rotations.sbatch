#!/bin/bash

#SBATCH --job-name=train_rotations
#SBATCH --open-mode=append
#SBATCH --output=logs/train_rotations_sched_%j.out
#SBATCH --error=logs/train_rotations_sched_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x 

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/ccp5804/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"

#cd $HOME/test
cd /scratch/ccp5804/DL21SP_final/rotations

singularity exec --nv \
--bind /scratch \
--overlay /scratch/ccp5804/DL21SP/conda.sqsh:ro \
--overlay /tmp/student_dataset.sqsh:ro \
/scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dev
python train_rotations.py --checkpoint-dir checkpoints/rotations_sched
"
