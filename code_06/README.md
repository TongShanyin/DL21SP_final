# Training Procedure #

* Train the encoder (ResNet50 after average pool layer) using SimCLR and 512,000 unlabeled data.

	We use TEMPERATURE = 0.1, use SGD with lr = 0.1 and learning rate scheduler and train the model for 200 epochs, takes about 80 hours in total.
	
	On GCP node, since the memory is limited, we can only use BATCH_SIZE = 256, and also the training time per task is limited by 24h, so we need to split the whole training into several tasks. For the first task, we need to run: 

		sbatch train_contrastive.sbatch

	The trained encoder will be stored in checkpoints/simclr_encoder.pth, the checkpoints for the SimCLR model will be stored in checkpoints/simclr_net.pth. After the first task, we want to use the previous checkpoints to continue training. First, we need to uncomment the corresponding codes in train_contrastive.py (already mark by comments). Then, we need change the line in train_contrastive.sbatch to be:

		python train_contrastive.py --checkpoint-dir checkpoints --checkpoint_net checkpoints/simclr_net.pth  

	And then use sbatch train_contrastive.sbatch to continue training. If we want to use larger batchsize like BATCH_SIZE = 1024, we need to train in Greene directly.

* Train the linear classfier (connected with the fronzen encoder from last step) using labeled data.  train_classifier.sbatch launches the training code in train_classifier.py.  Commenting/uncommenting lines 20 and 21 switch between training with extra data and without it.  We train for 300 epochs and choose the checkpoint with the best validation accuracy.  For us, that was epoch 49 without extra data and epoch 67 with extra data.

* Fine-tuning the whole model using the labeled data.  This requires plugging the classifier checkpoint into finetune.sbatch, which launches the finetuning script finetune.py.  Again, switching the value for trainset is used to add or remove extra data.  Our best results are at epoch 52 without extra data and epoch 76 with extra data.
