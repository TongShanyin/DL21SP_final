# DL21SP_final

## Autoencoder

The `autoencoder` directory contains code and training scripts for the unsupervised part of our model. `train_autoencoder.sbatch` trains the autoencoder on the unsupervised data and saves out a checkpoint into `checkpoints/`.  `train_classifier.sbatch` then loads the checkpoint and combines it with a dummy classifier (a small convolutional net) and trains on the supervised data.
