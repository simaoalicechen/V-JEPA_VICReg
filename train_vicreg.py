# only vicreg, jepa see train_jepa.py, train_jepa2.py, train_jepa3.py

"""
Part 1: (see train_vpred.py)
supervised learning on video data for prediction

1. download moving mnist 
2. dataloader and repackage
3. first_10_seconds --> inputs
4. second_10_seconds --> targets
5. train, val, test, train with ConvLSTM and MSELoss

Part 2: this file (train_vicreg.py)

self-supervised learning with vicreg on two inputs, original and augmented one, to go through 
2 identical encoders, either convLSTM or cnn_3layers. 

6. groups of inputs based on the time frames they are at, sequentially sent as inputs
7. train with convnet (encoder) and VICReg loss with the two embeddings
8. outputs = 2 sets of representations/embeddings
9. results are 2 sets of very small losses

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
from IPython.display import display
from torch.optim import Adam
from ipywidgets import HBox, Image as WImage
from conv_3Layer import CNN_3Layer
import ipywidgets as widgets
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import io
import imageio
from ipywidgets import widgets, HBox
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
from vicreg import vicreg_loss
import sys
import resnet
import wandb
import inspect
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# setups 
# hyperperameters
parser = argparse.ArgumentParser(description="vicreg_training_script")
parser.add_argument("--learning_rate", type=float, default=0.001, help="default learning rate for the optimizer") 
parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD", "RMSprop"])
parser.add_argument("--epochs", type= int, default = 10, help = "number of training epochs")
parser.add_argument("--batch_size", type = float, default = 32)
parser.add_argument("--weight_decay", type = float, default=1e-5)
parser.add_argument("--lr_scheduler", type = str, default = "none", choices=["none", "cosine", "step", "plateau"])

# loss weights for var, cov, and pred
parser.add_argument("--var_loss_weight", type=float, default=15.0 )
parser.add_argument("--cov_loss_weight", type=float, default=1.0)
parser.add_argument("--inv_loss_weight", type=float, default=25.0)


args = parser.parse_args()

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
        "model": "ConvNet",
        "ssl_method": "VICReg",
        "lambda_inv": args.inv_loss_weight,
        "lambda_var": args.var_loss_weight,
        "lambda_cov": args.cov_loss_weight,
    }
)

# Shuffle Data
np.random.shuffle(MovingMNIST)
train_data = MovingMNIST        

# use MovingMNISTDataset directly
class MovingMNISTDataset(Dataset):
    def __init__(self, data_path = 'mnist_test_seq.npy'):
        self.data = np.load(data_path).transpose(1, 0, 2, 3)
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        sequence = torch.FloatTensor(sequence).unsqueeze(1)
        return sequence/255.0

# corrected previous errors 
# not using augmented different version of inputs, but use odd frames as one set of inputs
# and even frames as another set of inputs
# just to work on a simple encoder and use the representations with vicreg
# train_jepa only uses variance and covariance on actual representations 
# and pred on predictions and actual representations
def forward_pass(epoch, batch_num, encoder1, encoder2, frames, args):
    frames = frames.to(device)
    seq_length = frames.shape[1]
    batch_size = frames.shape[0]

    odd_frames = frames[:, ::2]
    even_frames = frames[:, 1::2]

    min_length = min(odd_frames.shape[1], even_frames.shape[1])

    total_cov_loss = 0.0
    total_var_loss = 0.0
    total_inv_loss = 0.0
    total_vicreg_loss = 0.0
    for t in range(min_length):
        odd_representation = encoder1(odd_frames[:, t])
        even_representation = encoder2(even_frames[:, t])
        var_loss, cov_loss, inv_loss = vicreg_loss(odd_representation, even_representation)
        total_cov_loss += cov_loss
        total_var_loss += var_loss
        total_inv_loss += inv_loss 
        total_vicreg_loss += var_loss*args.var_loss_weight + cov_loss*args.var_loss_weight + \
                        inv_loss*args.inv_loss_weight

    batch_loss_dict = {
        "epoch": epoch, 
        "batch_num": batch_num,
        "total_var_loss": total_var_loss,
        "total_cov_loss": total_cov_loss, 
        "total_inv_loss": total_inv_loss, 
        # weights defined in the function file
        "total_vicreg_loss": total_vicreg_loss
    }

    return batch_loss_dict

def train_one_epoch(epoch, encoder1, encoder2, dataloader, optimizer): 
    encoder1.train()
    encoder2.train()
    total_loss = 0.0
    epoch_loss_dict = {
        "epoch": epoch, 
        "total_var_loss": 0.0,
        "total_cov_loss": 0.0,
        "total_inv_loss":0.0,
        "total_vicreg_loss": 0.0
    }
    total_batch_length = len(dataloader)
    for batch_num, batch in enumerate(dataloader):
        loss_dict = forward_pass(epoch, batch_num, encoder1, encoder2, batch, args)
        vicreg_loss = loss_dict["total_vicreg_loss"]
        optimizer.zero_grad()
        vicreg_loss.backward()
        optimizer.step()
        for key in epoch_loss_dict:
            if key != "epoch":
                epoch_loss_dict[key] += loss_dict[key]

    epoch_logged_losses = {
        "epoch": epoch, 
        "total_var_loss": epoch_loss_dict['total_var_loss'].item()/total_batch_length,
        "total_cov_loss": epoch_loss_dict['total_cov_loss'].item()/total_batch_length,
        "total_inv_loss": epoch_loss_dict['total_inv_loss'].item()/total_batch_length,
        "total_vicreg_loss": epoch_loss_dict['total_vicreg_loss'].item()/total_batch_length
    }

    print(epoch_logged_losses)

    wandb.log({
        "epoch": epoch_logged_losses['epoch'], 
        "total_var_loss": epoch_logged_losses['total_var_loss'], 
        "total_cov_loss": epoch_logged_losses['total_cov_loss'], 
        "total_inv_loss": epoch_logged_losses['total_inv_loss'], 
        "total_vicreg_loss": epoch_logged_losses['total_vicreg_loss']
    })

    return epoch_logged_losses
    
def val_one_epoch(epoch, encoder1, encoder2, val_loader):

    encoder1.eval()
    encoder2.eval()
    total_loss = 0.0 
    loss_dict_val ={}

    epoch_loss_dict_val = {
        "epoch": epoch, 
        "total_var_loss": 0.0,
        "total_cov_loss": 0.0,
        "total_inv_loss":0.0,
        "total_vicreg_loss": 0.0
    }
    total_batch_length = len(val_loader)

    with torch.no_grad():
        for batch_num, batch in enumerate(val_loader):
            loss_dict = forward_pass(epoch, batch_num, encoder1, encoder2, batch, args)
            for key in epoch_loss_dict_val:
                if key != "epoch":
                    epoch_loss_dict_val[key] += loss_dict[key]

    epoch_logged_losses_val = {
        "epoch": epoch, 
        "total_var_loss": epoch_loss_dict_val['total_var_loss'].item()/total_batch_length,
        "total_cov_loss": epoch_loss_dict_val['total_cov_loss'].item()/total_batch_length,
        "total_inv_loss": epoch_loss_dict_val['total_inv_loss'].item()/total_batch_length,
        "total_vicreg_loss": epoch_loss_dict_val['total_vicreg_loss'].item()/total_batch_length
    }

    post_fix = "_val"
    new_epoch_logged_losses_val = {key + post_fix: value for key, value in epoch_logged_losses_val.items()}
    print(new_epoch_logged_losses_val)
    wandb.log({
        "epoch_val": epoch_logged_losses_val['epoch'], 
        "total_var_loss_val": epoch_logged_losses_val['total_var_loss'], 
        "total_cov_loss_val": epoch_logged_losses_val['total_cov_loss'], 
        "total_inv_loss_val": epoch_logged_losses_val['total_inv_loss'], 
        "total_vicreg_loss_val": epoch_logged_losses_val['total_vicreg_loss']
    })

    return new_epoch_logged_losses_val


dataset = MovingMNISTDataset()
dataloader = DataLoader(dataset, batch_size = 8, shuffle = True)

print("after train loaders")
print(len(dataloader))

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [8000, 2000])
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True)
encoder1 = CNN_3Layer(input_channels=1, output_dim=256).to(device)
encoder2 = CNN_3Layer(input_channels=1, output_dim=256).to(device)
# optim = Adam(list(encoder1.parameters())+ list(encoder2.parameters()), lr=1e-3)

# arg parser stuff: 
learning_rate = args.learning_rate
optimizer_name = args.optimizer
num_epochs = args.epochs
var_loss_weight = args.var_loss_weight
cov_loss_weight = args.cov_loss_weight
inv_loss_weight = args.inv_loss_weight 


# todo
weight_decay = args.weight_decay
lr_scheduler = args.lr_scheduler

# args naming: 
if optimizer_name == "Adam":
    optim = Adam(list(encoder1.parameters())+ list(encoder2.parameters()), lr=learning_rate)
elif optimizer_name == "SGD":
    optim = torch.optim.SGD(list(encoder1.parameters())+ list(encoder2.parameters()), lr=learning_rate) 
elif optimizer_name == "RMSprop":
    optim = torch.optim.RMSprop(list(encoder1.parameters())+ list(encoder2.parameters()), lr=learning_rate)

print("no errors so far")

for epoch in range(1, num_epochs+1):
    train_loss_dict = train_one_epoch(epoch, encoder1, encoder2, train_loader, optim)
    val_loss_dict = val_one_epoch(epoch, encoder1, encoder2, val_loader)

wandb.finish()     
