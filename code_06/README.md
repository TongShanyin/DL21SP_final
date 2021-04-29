# Training Procedure #

* Train the encoder (ResNet50 after average pool layer) using SimCLR and 512,000 unlabeled data.

	sbatch train_contrastive.sbatch

* Train the linear classfier (connected with the fronzen encoder from last step) using labeled data. 

* Fine-tuning the whole model using the labeled data.
