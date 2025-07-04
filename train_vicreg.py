# based upon: https://pytorchvideo.org/docs/tutorial_torchhub_inference 
# and https://sladewinter.medium.com/video-frame-prediction-using-convlstm-network-in-pytorch-b5210a6ce582
# and https://pytorchvideo.org/docs/tutorial_classification

"""
Part 1: (see train.py)
supervised learning on video data for prediction

1. download moving mnist 
2. dataloader and repackage
3. first_10_seconds --> inputs
4. second_10_seconds --> targets
5. train, val, test, train with ConvLSTM and MSELoss

Part 2: this file (train2.py)

self-supervised learning with vicreg on two inputs, original and augmented one, to go through 
2 identical encoders, either convLSTM or cnn_3layers. 

6. group1 = original
7. group2 = augmented 
8. train with convnet (encoder) and VICReg loss with the two embeddings
9. outputs = 2 sets of representations/embeddings
10. results are 2 sets of very small losses

"""
'''
remember to run in terminal: export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
'''

import torch
import json
import time 
import argparse
from pathlib import Path
import os
from torchvision import transforms
# from torchvision.transforms import v2
from PIL import Image
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms import Compose, Lambda
from torch.utils.data import DataLoader
from pytorchvideo.data.encoded_video import EncodedVideo
from resnet import resnet50
from ConvLSTM import ConvLSTM 
from IPython.display import display
from ipywidgets import HBox, Image as WImage
import ipywidgets as widgets
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from conv_3Layer import CNN_3Layer
import torch.nn.functional as F
from Seq2Seq import Seq2Seq
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import io
import imageio
from ipywidgets import widgets, HBox
from skimage.metrics import structural_similarity as ssim
from vicreg import vicreg_loss
import sys
import resnet
import wandb
import inspect
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# load from npy
MovingMNIST = np.load('mnist_test_seq.npy').transpose(1, 0, 2, 3)
len(MovingMNIST)

#wandb
os.environ["WANDB_MODE"] = "offline"
wandb.init(
    project="ssl-vicreg-2",
    name="vicreg-LSTM-movingmnist",
    mode="offline", 
    config={
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "model": "CNN",
        "ssl_method": "VICReg",
        "lambda_inv": 25.0,
        "lambda_var": 25.0,
        "lambda_cov": 1.0
    }
)

# Shuffle Data
np.random.shuffle(MovingMNIST)
train_data = MovingMNIST        


def collate(batch):
    start_time = time.time()
    batch = torch.from_numpy(np.array(batch)).unsqueeze(1)
    batch = batch / 255.0                        
    batch = batch.to(device)        
    # the whole 20 frames are inputs to generate embeddings, z1         
    original_inputs = batch
    # augmented inputs are used for self supervised learning to generate 
    # another set of embeddings, z2
    aug_inputs = F.interpolate(
        original_inputs.reshape(-1, 1, 64, 64), 
        size = (224, 224), 
        mode = 'bilinear', 
        align_corners = False
    ).reshape(original_inputs.shape[0], 1, 10, 224, 224)
    # resize back to 64X64 so that we can feed to the same model
    aug_inputs = F.interpolate(
        aug_inputs.reshape(-1, 1, 224, 224), 
        size=(64, 64), 
        mode='bilinear', 
        align_corners=False
    ).reshape(original_inputs.shape[0], 1, 10, 64, 64)
    # transforms.RandomHorizontalFlip(p=0.5),
    # random horizontal flip 
    if torch.rand(1) > 0.5:
        aug_inputs = torch.flip(aug_inputs, dims=[4])

    load_time = time.time() - start_time       
    print(load_time)
    return original_inputs, aug_inputs

# Running for the first time, train and val dataloader creation: 
train_loader = DataLoader(train_data, shuffle=False, 
                        batch_size=8, collate_fn = collate)
print('training loader done')

save_dir = 'processed_batches'
os.makedirs(save_dir, exist_ok=True)

# if having run this once, no need to run again. They have been saved to the disk
current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'processed_batches')
print("processing and saving batches to disk")
batch_count = 0
for i, (original_inputs, aug_inputs) in enumerate(train_loader):
    batch_data = {
        'original_inputs': original_inputs.cpu(),
        'aug_inputs': aug_inputs.cpu()
    }
    torch.save(batch_data, f'{save_dir}/batch_{i:04d}.pt')
    del original_inputs, aug_inputs, batch_data
    torch.cuda.empty_cache()
    
    batch_count += 1

    if i%50 == 0:
        print(f"processing {i} batches")

'''
After running once, use saved data from the disk (saving sapce and time)
'''
class ProcessedMovingMNIST(torch.utils.data.Dataset):
    def __init__(self, data_path):

        self.data_path = data_path
        self.batch_files = sorted([f for f in os.listdir(data_path) if f.endswith('.pt')])

        first_batch_path = os.path.join(data_path, self.batch_files[0])
        first_data = torch.load(first_batch_path, map_location='cpu')
        
        self.batch_size = len(first_data['original_inputs'])
        self.length = len(self.batch_files) * self.batch_size

        self.original_shape = first_data['original_inputs'][0].shape
        self.aug_shape = first_data['aug_inputs'][0].shape

        del first_data

    def __len__(self):
        return self.length 

    def __getitem__(self, idx):
        batch_idx = idx // self.batch_size
        within_batch_idx = idx % self.batch_size
        batch_file = self.batch_files[batch_idx]
        batch_path = os.path.join(self.data_path, batch_file)
        data = torch.load(batch_path, map_location='cpu')
        original = data['original_inputs'][within_batch_idx].clone()
        aug = data['aug_inputs'][within_batch_idx].clone()

        del data

        return original, aug
        

saved_train_dataset = ProcessedMovingMNIST(save_dir)
train_loader = torch.utils.data.DataLoader(saved_train_dataset, batch_size = 32, shuffle = True)

# The input video frames are grayscale, thus single channel
# mainly for video encoder and decoder 
# model1 = Seq2Seq(num_channels=1, num_kernels=64, 
# kernel_size=(3, 3), padding=(1, 1), activation="relu", 
# frame_size=(64, 64), num_layers=3).to(device)
# model1.to(device)

# model2 = Seq2Seq(num_channels=1, num_kernels=64, 
# kernel_size=(3, 3), padding=(1, 1), activation="relu", 
# frame_size=(64, 64), num_layers=3).to(device)
# model2.to(device)

# a very simple cnn encoder
# two same models/encoder runing on two inpust to generate two sets of embeddings 
# for traning with vicreg 
model1 = CNN_3Layer(input_channels=1, output_dim=256).to(device)
model2 = CNN_3Layer(input_channels=1, output_dim=256).to(device)


# can be seperate with 2 different optimizers too
# if the same optimizer, contatenate the model parameters 
optim = Adam(list(model1.parameters())+ list(model2.parameters()), lr=1e-4)

num_epochs = 10

for epoch in range(1, num_epochs+1):
    
    train_loss = 0.0           
    var_loss = 0.0
    cov_loss = 0.0
    inv_loss = 0.0                              
    model1.train() 
    model2.train()   

    print(epoch)
    print("train")
    print(len(train_loader))
    for batch_num, (input1, input2) in enumerate(train_loader, 1):  
        # print(input1.dim())
        if batch_num%50==0:
            print(batch_num)

        if input1.dim() == 4:
            input1 = input1.unsqueeze(1)  
        if input2.dim() == 4:
            input2 = input2.unsqueeze(1)
            
        input1 = input1.to(device)
        input2 = input2.to(device)

        # two sets of output embeddings 
        z1 = model1(input1)      
        z2 = model2(input2)
        z1_flat = z1.view(z1.shape[0], -1)
        z2_flat = z2.view(z2.shape[0], -1)
        # print(z1.shape, z2.shape, "z1, z2, shape")
        loss, var, cov, inv = vicreg_loss(z1_flat, z2_flat)
        optim.zero_grad()
        loss.backward()
        optim.step()


        if z1.shape[1] == 1:
            z1 = z1.repeat(1, 3, 1, 1)
        if z2.shape[1] == 1:
            z2 = z2.repeat(1, 3, 1, 1)  

        train_loss += loss.item()
        var_loss += var.item()
        cov_loss += cov.item()
        inv_loss += inv.item()

        # wandb
        if batch_num % 50 == 0:
            wandb.log({
                "batch_total_loss": train_loss,
                "batch_variance_loss": var_loss,
                "batch_covariance_loss": cov_loss, 
                "batch_invariance_loss": inv_loss, 
                "epoch": epoch,
                "batch": batch_num,
                "learning_rate": optim.param_groups[0]['lr']
            })


        if batch_num % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_num}: '
                      f'Total loss: {train_loss:.4f}, '
                      f'Var: {var_loss:.4f}, '
                      f'Cov: {cov_loss:.4f}, '
                      f'Inv: {inv_loss:.4f}')


        del input1, input2, loss, var, cov, inv, 
        torch.cuda.empty_cache()               


    train_loss /= len(train_loader.dataset)    
    num_batches = len(train_loader)

    # wandb after an epoch
    wandb.log({
        "epoch": epoch,
        "epoch_total_loss": train_loss / num_batches,
        "epoch_variance_loss": var_loss / num_batches,
        "epoch_covariance_loss": cov_loss / num_batches,
        "epoch_invariance_loss": inv_loss / num_batches,
    })

    print(f'Epoch {epoch}: ', 
                      f'Total loss: {train_loss:.4f}, '
                      f'Var: {var_loss:.4f}, '
                      f'Cov: {cov_loss:.4f}, '
                      f'Inv: {inv_loss:.4f}') 

# finish after all runs
wandb.finish()            

"""
Result 2 with CNN_3Layer encoder and vicreg:
For full results, see wandb_results folder

Epoch 10, Batch 100: Total loss: 0.0002, Var: 0.0000, Cov: 0.0000, Inv: 0.0000
150
200
Epoch 10, Batch 200: Total loss: 0.0005, Var: 0.0000, Cov: 0.0000, Inv: 0.0000
250
300
Epoch 10, Batch 300: Total loss: 0.0007, Var: 0.0000, Cov: 0.0000, Inv: 0.0000
Epoch 10:  Total loss: 0.0000, Var: 0.0000, Cov: 0.0000, Inv: 0.0000
"""