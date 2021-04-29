# Training Procedure #

* Train the encoder (ResNet50 after average pool layer) using SimCLR and 512,000 unlabeled data.

	We use BATCH_SIZE = 1024, TEMPERATURE = 0.1, use SGD with lr = 0.1 and learning rate scheduler and train the model for 200 epochs, takes about 80 hours in total.

	* The easiest way to replicate our experiment is to use Greene. On Greene, first scp the DL21SP folder from the GCP node to /scratch/$USER, then run the command:

			sbatch greene_train_constractive.sbatch 

		The trained encoder will be stored in checkpoints/simclr_encoder.pth, the checkpoints for the SimCLR model will be stored in checkpoints/simclr_net.pth

	* The other way is to use GCP node directly. On GCP node, since the memory is limited, we can not use BATCH_SIZE = 1024. First, need to change BATCH_SIZE = 256 in train_contrastive.py, and also the training time per task is limited by 24h, for the first task, we need to run: 

			sbatch train_contrastive.sbatch

		After the first task, we want to use the previous checkpoints to continue training. First, we need to uncomment the corresponding codes in train_contrastive.py (already mark by comments). Then, we need change the line in train_contrastive.sbatch to be:

			python train_contrastive.py --checkpoint-dir checkpoints --checkpoint_net checkpoints/simclr_net.pth  

		And then use sbatch to run the file.

* Train the linear classfier (connected with the fronzen encoder from last step) using labeled data. 

* Fine-tuning the whole model using the labeled data.
