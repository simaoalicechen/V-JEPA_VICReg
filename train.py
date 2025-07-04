# based upon: https://pytorchvideo.org/docs/tutorial_torchhub_inference 
# and https://sladewinter.medium.com/video-frame-prediction-using-convlstm-network-in-pytorch-b5210a6ce582
# this is a straightforward supervised learning of video data prediction training 
# using ConvLSTM and MSELoss 

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
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
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
from Seq2Seq import Seq2Seq
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.nn.functional as F
import io
import imageio
# from conv_3Layer2 import CNNEncoderDecoder
from ipywidgets import widgets, HBox
from skimage.metrics import structural_similarity as ssim
import wandb
import sys
import resnet
import inspect
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# load from npy
MovingMNIST = np.load('mnist_test_seq.npy').transpose(1, 0, 2, 3)
len(MovingMNIST)

# Shuffle Data
np.random.shuffle(MovingMNIST)

# Train, Test, Validation splits
train_data = MovingMNIST[:8000]         
val_data = MovingMNIST[8000:9000]       
test_data = MovingMNIST[9000:10000] 
os.environ["WANDB_MODE"] = "offline"
wandb.init(
    project="video-prediction-CNN-mse",
    name="video-CNN-experiment-1",  
    mode = "offline", 
    config={
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "model": "CNN",
        "dataset": "MovingMNIST"
    }
)

# Peak Signal-to-Noise Ratio
# the quality of images/videos reconstructed 
def psnr(pred, target, max_val = 1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    psnr = 20*torch.log10(torch.tensor(max_val)/torch.sqrt(mse))
    return psnr.item()

def mae(pred, target):
    return F.l1_loss(pred, target).item()

# here
def collate(batch):
    start_time = time.time()
    batch = torch.from_numpy(np.array(batch)).unsqueeze(1)
    batch = batch / 255.0                        
    batch = batch.to(device)                     
    inputs = batch[:, :, :10]
    targets = batch[:, :, 10:]
    load_time = time.time() - start_time       
    # print(load_time)
    return inputs, targets  

'''
Running for the first time, train and val dataloader creation: 
'''
# train_loader = DataLoader(train_data, shuffle=False,
#                         batch_size=16, collate_fn = collate)
# print('training loader done')

# val_loader = DataLoader(val_data, shuffle=False, 
#                         batch_size=16, collate_fn = collate)
# print('val done') 

# all_inputs = []
# all_targets = []
# print("collecting")
# for i, (inputs, targets) in enumerate(train_loader):
#     all_inputs.append(inputs.cpu())
#     all_targets.append(targets.cpu())
#     if i%10 == 0:
#         print(f"processing {i} batches")
#     torch.cuda.empty_cache()
# print("concatenating all data ...")
# all_inputs = torch.cat(all_inputs, dim = 0)
# print("1")
# all_targets = torch.cat(all_targets, dim = 0)


# save_dir = "saved_train_data/"
# os.makedirs(save_dir, exist_ok = True)
# torch.save({
#     "inputs": all_inputs,
#     "targets": all_targets, 
# }, os.path.join(save_dir, 'train_data.pt'))

# print("saved")
# print(f"input shape: ", {all_inputs.shape})
# print(f'targest shape: ', {all_targets.shape})

# val_all_inputs = []
# val_all_targets = []
# print("collecting")
# for i, (inputs, targets) in enumerate(val_loader):
#     val_all_inputs.append(inputs.cpu())
#     val_all_targets.append(targets.cpu())
#     if i%10 == 0:
#         print(f"processing {i} val batches")
#     torch.cuda.empty_cache()
# print("concatenating all data ...")
# val_all_inputs = torch.cat(val_all_inputs, dim = 0)
# print("1")
# val_all_targets = torch.cat(val_all_targets, dim = 0)


# save_dir = "saved_val_data/"
# os.makedirs(save_dir, exist_ok = True)
# torch.save({
#     "inputs": val_all_inputs,
#     "targets": val_all_targets, 
# }, os.path.join(save_dir, 'val_data.pt'))



# if running for the first time 
# Get a batch
# input, _ = next(iter(val_loader))

# # Reverse process before displaying
# input = input.cpu().numpy() * 255.0     


# for i, video in enumerate(input.squeeze(1)[:3]):
#     with io.BytesIO() as gif:
#         imageio.mimsave(gif, video.astype(np.uint8), format="GIF", fps=5)
#         gif_bytes = gif.getvalue() 
#         display(HBox([WImage(value=gif_bytes)]))
#     with open(f"output_{i}.gif", "wb") as f:
#         f.write(gif_bytes)



'''
After running once, use saved data
'''
class ProcessedMovingMNIST(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path, map_location = 'cpu')
        self.inputs = data['inputs']
        self.targets = data['targets']
        self.length = len(self.inputs)

    def __len__(self):
        return self.length 

    def __getitem__(self, idx):
        return self.inputs[idx].clone(), self.targets[idx].clone()
        

saved_train_dataset = ProcessedMovingMNIST('saved_train_data/train_data.pt')
train_loader = torch.utils.data.DataLoader(saved_train_dataset , batch_size = 32, shuffle = True)
saved_val_dataset = ProcessedMovingMNIST('saved_val_data/val_data.pt')
val_loader = torch.utils.data.DataLoader(saved_val_dataset, batch_size = 32, shuffle = True)

# for batch_num, (input, target) in enumerate(train_loader, 1):
#     print(batch_num, input.shape, target.shape)
#     if batch_num >=3: 
#         break


# The input video frames are grayscale, thus single channel
model = Seq2Seq(num_channels=1, num_kernels=64, 
kernel_size=(3, 3), padding=(1, 1), activation="relu", 
frame_size=(64, 64), num_layers=3).to(device)
# model = resnet50()
model.to(device)
# model = CNNEncoderDecoder(input_channels=1, output_channels=1).to(device)

optim = Adam(model.parameters(), lr=1e-4)

criterion = nn.MSELoss()
criterion = criterion.to(device)
num_epochs = 10

for epoch in range(1, num_epochs+1):
    
    train_loss = 0.0           
    epoch_psnr = 0.0
    epoch_mae = 0.0                               
    model.train()     
    print(epoch)
    print("train")
    print(len(train_loader))
    for batch_num, (input, target) in enumerate(train_loader, 1):  
        if batch_num%100==0:
            print(batch_num)
        # print(batch_num)
        if input.dim() == 4:
            input = input.unsqueeze(1)  
        # print(f"Fixed input shape: {input.shape}")
        input = input.to(device)
        target = target.to(device)
        # print("train")
        optim.zero_grad() 
        output = model(input)      

        if output.shape[1] == 1:
            output = output.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)  

        loss = criterion(output, target)      
        # loss =  
        loss.backward()                                            
        optim.step()   
        with torch.no_grad():
            psnr = psnr(output, target)           
            mae = mae(output, target)                                                                          
        train_loss += loss.item()   
        epoch_psnr += psnr
        epoch_mae += mae    

        if batch_num %50 ==0:
            wandb.log({
                "batch_loss": loss.item(),
                "batch_psnr": psnr,
                "batch_mae": mae,
                "epoch": epoch,
                "batch": batch_num
            })
        print(f"Epoch {epoch}, Batch {batch_num}: "
                  f"Loss={loss.item():.4f}, PSNR={psnr:.2f}, MAE={mae:.4f}")

        del input, target, output, loss, psnr, mae
        torch.cuda.empty_cache()                         
    train_loss /= len(train_loader.dataset) 
    num_batches = len(train_loader)
    wandb.log({
        "epoch": epoch,
        "epoch_loss": train_loss,
        "epoch_psnr": epoch_psnr / num_batches,
        "epoch_mae": epoch_mae / num_batches,
        "learning_rate": optimizer.param_groups[0]['lr']
    })      

    print(f"Epoch {epoch} Summary:")
    print(f"Loss: {train_loss / num_batches:.6f}")
    print(f"PSNR: {epoch_psnr / num_batches:.2f} dB")
    print(f"MAE: {epoch_mae / num_batches:.4f}")

    val_loss = 0                                                 
    model.eval()            
    print("val")                                       
    with torch.no_grad():                                          
        for input, target in val_loader:    
            if input.dim() == 4:
                input = input.unsqueeze(1)
            input = input.to(device)
            target = target.to(device)                          
            output = model(input)  
            assert output.shape == target.shape
            if output.shape[1] == 1:
                output = output.repeat(1,3,1,1)    
            if target.shape[1] == 1:
                target= target.repeat(1,3,1,1)       

            loss = criterion(output, target)   
            val_loss += loss.item()             
            del input, target, output, loss 
            torch.cuda.empty_cache()                     
    val_loss /= len(val_loader.dataset)                            

    print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
        epoch, train_loss, val_loss))
wandb.finish()
# training loss: usually 0.04, validation loss: 0.04