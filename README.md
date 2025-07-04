This repo consists of 2 training objectives: 

 - Use ConvLSTM, MSE_loss, and Moving MNIST video data for frame prediction.
 - Use Conv 3 layer encoder, VICReg, and Moving MNIST video data to generate representations for self-supervised learning. 

Instructions:

conda env create -f environment.yml
conda activate video_training

VICReg results:

 - See wandb_results