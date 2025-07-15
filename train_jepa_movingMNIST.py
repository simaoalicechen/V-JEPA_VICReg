"""
These three things can happen simutaneously: (encoder forwarded embeddings/representations)
St, St+1, St+2 should be computed by (the same) encoders and VC loss
These three things can happen sequentially: (predictor forwarded embeddings/representations
Because they were the training results from 
St+1pred and St+1, St+2Pred and St+2 with training and L2 loss)
St_pred, St+1_pred, St+2_pred are from the (the same) predictors

All embeddings generated are just for practice and can be discarged
The kept ones are the encoder and predictor (world model)
"""
import torch
import argparse
import json
import time 
import cv2
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
from torch.optim import Adam, SGD
from conv_3Layer import CNN_3Layer
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
from predictor_with_action import Predictor_with_action

# setups 
# hyperparameter
parser = argparse.ArgumentParser(description="jepa_training_script")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="default learning rate for the optimizer") 
parser.add_argument("--action", type=str, default="even frames", choices=["even frames", None], help="currently, only even frames implemented as action")
parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD", "RMSprop"])
parser.add_argument("--epochs", type= int, default = 10, help = "number of training epochs")
parser.add_argument("--batch_size", type = float, default = 32)
parser.add_argument("--weight_decay", type = float, default=1e-5)
parser.add_argument("--lr_scheduler", type = str, default = "none", choices=["none", "cosine", "step", "plateau"])

# loss weights for var, cov, and pred
parser.add_argument("--var_loss_weight", type=float, default=1.0)
parser.add_argument("--cov_loss_weight", type=float, default=1.0)
parser.add_argument("--pred_loss_weight", type=float, default=10.0)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#wandb
os.environ["WANDB_MODE"] = "offline"
wandb.init(
    project="ssl-vicreg-jepa",
    name="v-jepa-movingMNIST",
    mode="offline", 
    config={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optim_choice": args.optimizer,
        "action_choice": args.action, 
        "var_loss_weight": args.var_loss_weight,
        "cov_loss_weight": args.cov_loss_weight,
        "pred_loss_weight": args.pred_loss_weight,
        "model": "conv_encover_predictor",
        "ssl_method": "VC + MSE",
    }
)

# directly read the moving mnist data
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

print("after loss functions")
"""
Since movingMNIST datasets are fixed ones, we can't influence their movements. 
However, we can treat the even frames as the actions.
Think of them as we probed the digits to suddenly move to the right/left as in the next frames
so that they can be considered as actions. 
"""
def forward_pass(epoch, batch_num, encoder, predictor, frames, args):
    # 20 --> seq_length 
    # batch = batch.to(device) 32 usually 
    frames = frames.to(device)
    # print("all frames dimensions: ", len(frames), frames.shape)
    if args.action == "even frames":
        predictor = Predictor_with_action().to(device)
        seq_length = frames.shape[1]
        batch_size = frames.shape[0]

        # keep all the first dimension, but skip every other row in the second dimensions
        # the other dimensions are intact (implicitly), because they were not mentioned and therefore
        # unchanged
        odd_frames = frames[:, ::2]
        even_frames = frames[:, 1::2]

        # the min_length only touches on the second dimension that was being sliced. 
        # the smallest one for both sets of frames would be the same in this case, 
        # because we have 20 frames
        # if we had 19 frames, the odd frames would be one more than the even frames. 
        min_length = min(odd_frames.shape[1], even_frames.shape[1])
        actual_representations = []
        action_representations = []
        total_cov_loss = 0.0
        total_var_loss = 0.0
        total_vc_loss = 0.0
        
        # encode all the inputs from odd frames, and slice out the actions from even frames
        # then flatten the action frames so they can be pairde with to be contantenated 
        # later (in the next for loop) with actual reprs to be sent to the predictor
        # in this loop, we also extract the VC loss from the actual reprs
        for t in range(min_length):
            represenation = encoder(odd_frames[:, t])
            action_frames = even_frames[:, t]
            actual_representations.append(represenation)
            # flatten action frames so that they can be concatenated with actual represenations
            # and then we can send the concatenated one to the predictor 
            # flatten from the second dimmension 
            flattened_action_frames = action_frames.flatten(start_dim = 1)
            action_representations.append(flattened_action_frames)
            var_loss = variance_loss(represenation)
            cov_loss = covariance_loss(represenation)
            total_cov_loss += cov_loss
            total_var_loss += var_loss
            total_vc_loss += (cov_loss + var_loss)
        total_pred_loss = 0.0
        total_weighted_loss = 0.0
        predicted_representations = []
        current_representation = actual_representations[0]
        current_action = action_representations[0]
        # with only 9 rounds here in this loop
        for t in range(min_length-1):
            # The issue was that actual presentations and action ones had different 
            # dimensions. If we send the actions to the encoder, it would be 
            # as if we just processed them together
            # One way to do is to concatenate them: together_rpr = torch.cat((represenation, flattened_action_frames), dim = 1) 
            # but we moved this in the predictor to be processed as different branches 
            predicted_representation = predictor(current_representation, current_action)
            pred_loss = pred_loss_computations(predicted_representation, actual_representations[t+1])
            total_pred_loss += pred_loss
            current_representation = actual_representations[t+1]

    elif args.action == None: 
        predictor = Predictor().to(device)
        seq_length = frames.shape[1]
        batch_size = frames.shape[0]

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
            # current_representation = predicted_representation
            current_representation = actual_representations[t+1]

    total_weighted_loss = total_pred_loss*args.pred_loss_weight + total_var_loss*args.var_loss_weight + \
                            total_cov_loss*args.cov_loss_weight
    batch_loss_dict = {
            "epoch": epoch, 
            "batch_num": batch_num,
            "total_var_loss": total_var_loss,
            "total_cov_loss": total_cov_loss, 
            "total_vc_loss": total_vc_loss, 
            "total_pred_loss": total_pred_loss, 
            "total_weighted_loss": total_weighted_loss
        }
        # print("inside forward pass: ", batch_loss_dict)
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
        loss_dict = forward_pass(epoch, batch_num, encoder, predictor, batch, args)
        # weighted loss for backtracking
        # TODO, should discuss
        # print("in train one epoch function ")
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
        # "epoch": epoch_logged_losses['epoch'], 
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
            loss_dict = forward_pass(epoch, batch_num, encoder, predictor, batch, args)  
            # print(loss_dict)
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
        # "epoch_val": epoch_logged_losses_val['epoch'], 
        "total_var_loss_val": epoch_logged_losses_val['total_var_loss'], 
        "total_cov_loss_val": epoch_logged_losses_val['total_cov_loss'], 
        "total_vc_loss_val": epoch_logged_losses_val['total_vc_loss'], 
        "total_pred_loss_val": epoch_logged_losses_val['total_pred_loss'], 
        "total_weighted_loss_val": epoch_logged_losses_val['total_weighted_loss']
    })

    return new_epoch_logged_losses_val

dataset = MovingMNISTDataset()
dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

print("after train loaders")
print(len(dataloader))

# print(len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [8000, 2000])
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True)
encoder = Encoder().to(device)

# predictor is now depending on the action choice made
if args.action == "even frames":
    predictor = Predictor_with_action().to(device)
elif args.action == None:
    predictor = Predictor().to(device)
# optim = Adam(list(encoder.parameters())+ list(predictor.parameters()), lr=0.005)

# arg parser stuff: 
learning_rate = args.learning_rate
optimizer_name = args.optimizer
num_epochs = args.epochs
var_loss_weight = args.var_loss_weight
cov_loss_weight = args.cov_loss_weight
pred_loss_weight = args.pred_loss_weight 


# TODO to make the training have wider range of hyperparameter choices 
weight_decay = args.weight_decay
lr_scheduler = args.lr_scheduler

# args naming: 
if optimizer_name == "Adam":
    optim = Adam(list(encoder.parameters())+ list(predictor.parameters()), lr=learning_rate)
elif optimizer_name == "SGD":
    optim = torch.optim.SGD(list(encoder.parameters())+ list(predictor.parameters()), lr=learning_rate) 
elif optimizer_name == "RMSprop":
    optim = torch.optim.RMSprop(list(encoder.parameters())+ list(predictor.parameters()), lr=learning_rate)

print("no errors so far")

for epoch in range(1, num_epochs+1):
    print("predictor chosen: ", predictor)
    train_loss_dict = train_one_epoch(epoch, encoder, predictor, train_loader, optim)
    val_loss_dict = val_one_epoch(epoch, encoder, predictor, val_loader)

wandb.finish()

