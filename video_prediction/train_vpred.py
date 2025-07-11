# based upon: https://pytorchvideo.org/docs/tutorial_torchhub_inference 
# and https://sladewinter.medium.com/video-frame-prediction-using-convlstm-network-in-pytorch-b5210a6ce582
# this is a straightforward supervised learning of video data prediction training 
# using ConvLSTM and MSELoss 

"""
Part 1: see this file, train_vpred.py
supervised learning on video data for prediction

1. download moving mnist 
2. dataloader and repackage
3. first_10_seconds --> inputs
4. second_10_seconds --> targets
5. train, val, test, train with ConvLSTM and MSELoss

Part 2: see train_vicreg.py

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
from VAutoencoder import Autoencoder, Decoder, Encoder
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
print("start")
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

# psnr
# peak signal-to-noise ratio
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
    # first 10 frames training, second 10 frames as target/pred        
    # ideally, we should try using the first 19 frames in training data (input and output)
    # and the last one frame as target       
    # due to memory issues (or maybe something else) only use the first 10 frames here.
    inputs = batch[:, :, :10]
    targets = batch[:, :, 10]
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
# print("2")
# val_all_targets = torch.cat(val_all_targets, dim = 0)

# print("3")
# save_dir = "saved_val_data/"
# os.makedirs(save_dir, exist_ok = True)
# torch.save({
#     "inputs": val_all_inputs,
#     "targets": val_all_targets, 
# }, os.path.join(save_dir, 'val_data.pt'))

# # if running for the first time 
# # Get a batch
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
# print("4")


'''
After running once, use saved data
'''
class ProcessedMovingMNIST(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path, map_location = 'cpu')
        print("11")
        self.inputs = data['inputs']
        self.targets = data['targets']
        self.length = len(self.inputs)

    def __len__(self):
        return self.length 

    def __getitem__(self, idx):
        return self.inputs[idx].clone(), self.targets[idx].clone()


saved_train_dataset = ProcessedMovingMNIST('saved_train_data/train_data.pt')
print("5")
train_loader = torch.utils.data.DataLoader(saved_train_dataset , batch_size = 1, shuffle = True)
print("6")
saved_val_dataset = ProcessedMovingMNIST('saved_val_data/val_data.pt')
print("7")
val_loader = torch.utils.data.DataLoader(saved_val_dataset, batch_size = 1, shuffle = True)
print("8")
print("9")

# if 'model' in locals():
#     del model
# if 'optim' in locals():
#     del optim
# torch.cuda.empty_cache()

# The input video frames are grayscale, thus single channel
model = Seq2Seq(num_channels=1, num_kernels=64, 
kernel_size=(3, 3), padding=(1, 1), activation="relu", 
frame_size=(64, 64), num_layers=3).to(device)
# model = Autoencoder()
# model = Autoencoder(Encoder(), Decoder())
# model = resnet50()
model.to(device)
# model = CNNEncoderDecoder(input_channels=1, output_channels=1).to(device)

optim = Adam(model.parameters(), lr=1e-4)

criterion = nn.MSELoss()
# criterion = torch.nn.MSELoss() 
# criterion = nn.BCELoss(reduction='sum')
criterion = criterion.to(device)
num_epochs = 10

model = model.to(device)

def reset_batchnorm(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
            # Reset parameters too
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            print(f"Reset BatchNorm: {module}")

# Apply to your model
reset_batchnorm(model)

for epoch in range(1, num_epochs+1):          
    epoch_psnr = 0.0
    epoch_mae = 0.0 
    train_loss = 0                                                 
    model.train()                                                  
    for batch_num, (input, target) in enumerate(train_loader, 1):  
        for module in model.modules():
            if hasattr(module, 'reset_hidden_state'):
            module.reset_hidden_state()
        # Clear any cached states
        if hasattr(module, 'hidden_state'):
            module.hidden_state = None
        if hasattr(module, 'cell_state'):
            module.cell_state = None
        # if batch_num %50 == 0:
        #     print("batch_num: ", batch_num)
        input = input.to(device)
        print(input.shape)
        target = target.to(device)
        output = model(input)    
        print(output)
        output = output.to(device)                                 
        loss = criterion(output.flatten(), target.flatten())       
        loss.backward()                                            
        optim.step()                                               
        optim.zero_grad()        
        with torch.no_grad():
            psnr_value = psnr(output, target)           
            mae_value = mae(output, target)                                                                          
        train_loss += loss.item()   
        epoch_psnr += psnr_value
        epoch_mae += mae_value

        if batch_num %50 ==0:
            wandb.log({
                "batch_loss": loss.item(),
                "batch_psnr": psnr_value,
                "batch_mae": mae_value, 
                "epoch": epoch,
                "batch": batch_num
            })
            print(f"Epoch {epoch}, Batch {batch_num}: "
                    f"Loss={loss.item():.4f} ")   
        del input, target, output, loss 
        torch.cuda.empty_cache()                                  
        # train_loss += loss.item()             

    num_batches = len(train_loader)
    train_loss /= num_batches
    epoch_psnr /= num_batches
    epoch_mae /= num_batches

    wandb.log({
        "epoch": epoch,
        "epoch_loss": train_loss,
        "epoch_psnr": epoch_psnr,
        "epoch_mae": epoch_mae,
        "learning_rate": optim.param_groups[0]['lr']
    })      

    print(f"epoch {epoch} metrics:")
    print(f"loss: {train_loss :.6f}")
    print(f"PSNR: {epoch_psnr :.4f} dB")
    print(f"mae: {epoch_mae :.4f}")
     
    val_psnr = 0.0        
    val_mae = 0.0                    

    val_loss = 0                                                 
    model.eval()                                                   
    with torch.no_grad():                                          
        for input, target in val_loader:   
            input = input.to(device)
            target = target.to(device)                       
            output = model(input)          
            output = output.to(device)                         
            loss = criterion(output.flatten(), target.flatten())   
            val_loss += loss.item()     
            val_psnr += psnr(output, target)
            val_mae += mae(output, target)    

            del input, target, output, loss 
            torch.cuda.empty_cache()        
    val_batches = len(val_loader)
    val_loss /= val_batches
    val_psnr /= val_batches
    val_mae /= val_batches             

    wandb.log({
        "epoch_val": epoch,
        "epoch_train_loss": train_loss,
        "val_loss": val_loss, 
        "val_psnr": val_psnr, 
        "val_mae": val_mae
    })               

    print(f"Epoch:{epoch} Training Loss:{train_loss:.4f} Validation Loss:{val_loss:.4f}\n")
    print(f'val_psnr = {val_psnr: .4f}, val_mae={val_mae: .4f}')
wandb.finish()                         