This repo consists of 2 main training objectives: 

 - Build a V-JEPA (with VC loss) to train the conv encoder and predictor. 
 - Use a 3-layer conv encoder, VICReg, and Moving MNIST video data to generate representations for self-supervised learning.

and a video_prediction training task at video_prediction folder: 

 - Use ConvLSTM, MSE_loss, and Moving MNIST video data for frame prediction.

Instructions:

 - conda env create -f environment.yml (more complete) or conda env create -f environment2.yml
 - conda activate OpenSTL_fixed

Sample command lines: 
 - python *.py
 - python train_jepa.py --learning_rate 0.0001 --optimizer "Adam" --var_loss_weight 1 --cov_loss_weight 1 --pred_loss_weight 10 --epochs 10 
 - python train_vicreg.py --learning_rate 0.0001 --optimizer "Adam" --var_loss_weight 15 --cov_loss_weight 1 --inv_loss_weight 25 --epochs 1 

V-JEPA and VICReg results:

 - See wandb_results
 - result chart naming conventions: 
    - jepa _ 'epoch number' _ 'learning rate' _ 'var loss weight' _ 'cov loss weight' _ 'pred loss weight'_ 'Action param' _ 'Static Noise param' _ 'Static Noise Speed param' _ 'Noise param'
    - vicreg _ 'epoch number' _ 'learning rate' _ 'var loss weight' _ 'cov loss weight' _ 'inv loss weight'
