This repo consists of 2 main training objectives: 

 - Build a V-JEPA (with VC loss) to train the conv encoder and predictor. 
 - Use a 3-layer conv encoder, VICReg, and Moving MNIST video data to generate representations for self-supervised learning.

and a video_prediction training task at video_prediction folder: 

 - Use ConvLSTM, MSE_loss, and Moving MNIST video data for frame prediction.

Instructions:

 - conda env create -f environment.yml (more complete) or conda env create -f environment2.yml
 - conda activate OpenSTL_fixed

V-JEPA and VICReg results:

 - See wandb_results
