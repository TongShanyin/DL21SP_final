Checkpoints can be downloaded from GCP with the following command:

gsutil cp gs://ckpts_dl/<CKPT> .

The checkpoints names:

accuracy 22%: 

	(encoder) simclr_resnet50_norm_start_1024t01sgd1_encoder_ep_50.pth

	(classifier) classifier_simclr_resnet50_norm_start_1024t01sgd1_encoder_ep_50_epoch50.pth
	   
accuracy 24%: 

	(encoder) simclr_resnet50_norm_start_1024t01sgd1_encoder_ep_70.pth
	
accuract 27.7%:

	(encoder) simclr_resnet50_norm_start80_1024t01_sgd5e3_encoder.pth
