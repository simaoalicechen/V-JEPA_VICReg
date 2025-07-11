# did not converge at learning rate of 0.04
# weights: 0.001, 1, 1 for pred, var, and cov losses 
"""
These three things can happen simutaneously: (encoder forwarded embeddings/representations)
St, St+1, St+2 should be computed by (the same) encoders and VC loss
These three things can happen sequentially: (predictor forwarded embeddings/representations
Because they were the training results from 
St+1pred and St+1, St+2Pred and St+2 with training and L2 loss)
St_pred, St+1_pred, St+2_pred are from the (the same) predictors

All embeddings generated are just practice and can be discarged
The kept ones are the encoder and predictor (world model)
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
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms import Compose, Lambda
from torch.utils.data import DataLoader
from pytorchvideo.data.encoded_video import EncodedVideo
from resnet import resnet50
from IPython.display import display
from ipywidgets import HBox, Image as WImage
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
from skimage.metrics import structural_similarity as ssim
import sys
import resnet
import wandb
import inspect
from encoder import Encoder
from predictor import Predictor 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#wandb
os.environ["WANDB_MODE"] = "offline"
wandb.init(
    project="ssl-vicreg-jepa",
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
'''
After running once, use saved data from the disk (saving sapce and time)
'''
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

# loss functions
def pred_loss_computations(z1, z2):
    pred_loss = sum(F.mse_loss(pred, real) for pred, real in zip(z1, z2))
    return pred_loss

def variance_loss(actual_repr, gamma = 1):
    actual_repr = actual_repr - actual_repr.mean(dim=0)
    std_p = actual_repr.std(dim=0)
    var_loss = F.relu(gamma - std_p).mean()
    return var_loss

def covariance_loss(actual_repr):
    actual_repr = actual_repr - actual_repr.mean(dim=0)
    cov = (actual_repr.T @ actual_repr) / (actual_repr.shape[0] - 1)
    cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / actual_repr.shape[1]
    return cov_loss

# def cov_var_loss(actual_repr):
#     batch_size = actual_repr[0]
#     num_features = actual_repr[-1]

print("after loss functions")

def forward_pass(epoch, batch_num, encoder, predictor, frames, actions = None):
    # should be 20
    # batch = batch.to(device)
    frames = frames.to(device)
    seq_length = frames.shape[1]
    batch_size = frames.shape[0]
    # print("inside forward pass: seq_length: ", seq_length)
    # print('inside forward pass: batch size: ', batch_size)

    actual_representations = []
    total_cov_loss = 0.0
    total_var_loss = 0.0
    total_vc_loss = 0.0
    for t in range(seq_length):
        # one batch of all inputs' frames at time t
        represenation = encoder(frames[:, t])
        actual_representations.append(represenation)
        var_loss = variance_loss(represenation)
        cov_loss = covariance_loss(represenation)
        total_cov_loss += cov_loss
        total_var_loss += var_loss
        total_vc_loss += (cov_loss + var_loss)

    total_pred_loss = 0.0
    predicted_representations = []
    # through predictor, each actual representation will be converted to predicted_representation
    # each newly formed predicted representation will be added to prediction_representations
    current_representation = actual_representations[0]
    # seq_length-1 here, because the last St+n does not produce predictions any more
    for t in range(seq_length - 1):
        # action needs to be added here. 
        # one batch of all predictions' frames at time t
        predicted_representation = predictor(current_representation)
        # predicted_representations.append(predicted_representation)
        # all actual representations were formed and collected, so that we 
        # can just take one (the t+1 one) out and then calculate the loss between 
        # St+1 and St+1_pred
        pred_loss = pred_loss_computations(predicted_representation, actual_representations[t+1])
        total_pred_loss += pred_loss
        # use prediction for next step, not actual
        # still in this loop, so it will become the next frame sent to predictor. 
        current_representation = predicted_representation

    batch_loss_dict = {
        "epoch": epoch, 
        "batch_num": batch_num,
        "total_var_loss": total_var_loss,
        "total_cov_loss": total_cov_loss, 
        "total_vc_loss": total_vc_loss, 
        "total_pred_loss": total_pred_loss, 
        "total_weighted_loss": None
    }
    # based on the non-weighted loss and adjusted the weights accordingly at the learning rate 
    # of 0.005
    batch_loss_dict['total_weighted_loss'] = batch_loss_dict["total_pred_loss"]*0.001 + \
                        batch_loss_dict["total_var_loss"]*1 + batch_loss_dict["total_cov_loss"]*1
     
    # return predictions and corresponding targets 
    return batch_loss_dict

def train_one_epoch(epoch, encoder, predictor, dataloader, optimizer): 
    encoder.train()
    predictor.train()
    total_loss = 0.0
    epoch_loss_dict = {
        "epoch": epoch, 
        "total_var_loss": 0.0,
        "total_cov_loss": 0.0,
        "total_vc_loss":0.0,
        "total_pred_loss": 0.0,
        "total_weighted_loss": 0.0
    }
    total_batch_length = len(dataloader)
    for batch_num, batch in enumerate(dataloader):
        loss_dict = forward_pass(epoch, batch_num, encoder, predictor, batch)
        # weighted loss for backtracking
        # should disscuss (todo)
        total_weighted_loss = loss_dict["total_weighted_loss"]
        optimizer.zero_grad()
        total_weighted_loss.backward()
        optimizer.step()
        for key in epoch_loss_dict:
            if key != "epoch":
                epoch_loss_dict[key] += loss_dict[key]

    # only when logging, use .item() to show the numbers
    # in all other dictionary places, pass along all the items so it can backtrack
    epoch_logged_losses = {
        "epoch": epoch, 
        "total_var_loss": epoch_loss_dict['total_var_loss'].item()/total_batch_length,
        "total_cov_loss": epoch_loss_dict['total_cov_loss'].item()/total_batch_length,
        "total_vc_loss": epoch_loss_dict['total_vc_loss'].item()/total_batch_length,
        "total_pred_loss": epoch_loss_dict['total_pred_loss'].item()/total_batch_length,
        "total_weighted_loss": epoch_loss_dict['total_weighted_loss'].item()/total_batch_length
    }

    print(epoch_logged_losses)

    wandb.log({
        "epoch": epoch_logged_losses['epoch'], 
        "total_var_loss": epoch_logged_losses['total_var_loss'], 
        "total_cov_loss": epoch_logged_losses['total_cov_loss'], 
        "total_vc_loss": epoch_logged_losses['total_vc_loss'], 
        "total_pred_loss": epoch_logged_losses['total_pred_loss'], 
        "total_weighted_loss": epoch_loss_dict['total_weighted_loss']
    })

    return epoch_logged_losses
    
def val_one_epoch(epoch, encoder, predictor, val_loader):

    encoder.eval()
    predictor.eval()
    total_loss = 0.0 
    loss_dict_val ={}

    epoch_loss_dict_val = {
        "epoch": epoch, 
        "total_var_loss": 0.0,
        "total_cov_loss": 0.0,
        "total_vc_loss":0.0,
        "total_pred_loss": 0.0,
        "total_weighted_loss": 0.0
    }
    total_batch_length = len(val_loader)

    with torch.no_grad():
        for batch_num, batch in enumerate(val_loader):
            loss_dict = forward_pass(epoch, batch_num, encoder, predictor, batch)
            total_weighted_loss = loss_dict["total_weighted_loss"]
            for key in epoch_loss_dict_val:
                if key != "epoch":
                    epoch_loss_dict_val[key] += loss_dict[key]

    # only when logging, use .item() to show the numbers
    # in all other dictionary places, pass along all the items so it can backtrack
    epoch_logged_losses_val = {
        "epoch": epoch, 
        "total_var_loss": epoch_loss_dict_val['total_var_loss'].item()/total_batch_length,
        "total_cov_loss": epoch_loss_dict_val['total_cov_loss'].item()/total_batch_length,
        "total_vc_loss": epoch_loss_dict_val['total_vc_loss'].item()/total_batch_length,
        "total_pred_loss": epoch_loss_dict_val['total_pred_loss'].item()/total_batch_length,
        "total_weighted_loss": epoch_loss_dict_val['total_weighted_loss'].item()/total_batch_length
    }

    # epoch_logged_losses_val.rename()
    post_fix = "_val"
    new_epoch_logged_losses_val = {key + post_fix: value for key, value in epoch_logged_losses_val.items()}
    print(new_epoch_logged_losses_val)
    wandb.log({
        "epoch_val": epoch_logged_losses_val['epoch'], 
        "total_var_loss_val": epoch_logged_losses_val['total_var_loss'], 
        "total_cov_loss_val": epoch_logged_losses_val['total_cov_loss'], 
        "total_vc_loss_val": epoch_logged_losses_val['total_vc_loss'], 
        "total_pred_loss_val": epoch_logged_losses_val['total_pred_loss'], 
        "total_weighted_loss_val": epoch_logged_losses_val['total_weighted_loss']
    })

    return new_epoch_logged_losses_val

dataset = MovingMNISTDataset()
dataloader = DataLoader(dataset, batch_size = 8, shuffle = True)

print("after train loaders")
print(len(dataloader))

# print(len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [8000, 2000])
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True)
encoder = Encoder().to(device)
predictor = Predictor().to(device)
optim = Adam(list(encoder.parameters())+ list(predictor.parameters()), lr=0.004)

num_epochs = 20

print("no errors so far")

for epoch in range(1, num_epochs+1):
    train_loss_dict = train_one_epoch(epoch, encoder, predictor, train_loader, optim)
    val_loss_dict = val_one_epoch(epoch, encoder, predictor, val_loader)

wandb.finish()

# odd number of frames as inputs, and even number of frames as actions