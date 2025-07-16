import torch
import argparse
import json
import time 
import argparse
from pathlib import Path
from enum import Enum, auto
import torchvision
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms import Compose, Lambda
from torch.utils.data import DataLoader
from pytorchvideo.data.encoded_video import EncodedVideo
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
import wandb
import inspect
from encoder import Encoder
from predictor import Predictor 
# from predictor_with_moving_dot import Predictor_md
from predictor_dot_action import Predictor_with_action
from single import ContinuousMotionDataset, DeterministicMotionDataset
from multiple import MultiDotDataset, create_three_datasets

# setups 
# hyperparameter
parser = argparse.ArgumentParser(description="jepa_training_script")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="default learning rate for the optimizer") 
# disable this for the moving dot, because the action will be determined by the batch's properties
parser.add_argument("--action", type=str, default= None, choices=["Yes", None], help="currently, action (a string) just means we will send the action vector to the predictor")
parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD", "RMSprop"])
parser.add_argument("--epochs", type= int, default = 10, help = "number of training epochs")
parser.add_argument("--batch_size", type = float, default = 32)
parser.add_argument("--weight_decay", type = float, default=1e-5)
parser.add_argument("--lr_scheduler", type = str, default = "none", choices=["none", "cosine", "step", "plateau"])

# loss weights for var, cov, and pred
parser.add_argument("--var_loss_weight", type=float, default=1.0)
parser.add_argument("--cov_loss_weight", type=float, default=1.0)
parser.add_argument("--pred_loss_weight", type=float, default=10.0)

# settings for input videos
parser.add_argument("--concentration", type = float, default = 0.2, help = "the lower the number, the more unpredictable the actions would be")

# static noise and how they change depend on the static noise setting (the lower the less noise)
# and the speed. Once there is speed, the frames will have sequentially changed static noise on them 
# aka, the noise are no longer static for all frames
parser.add_argument("--static_noise", type = float, default= 0, help = "the lower the number, the less noise would be in overlay (but they would be the same noise for all frames if the speed is 0)")
parser.add_argument("--static_noise_speed", type = float, default= 0, help = "the higher the absolute value of the number, the more rollover of the static noise patterns woudl be, with torch.roll()")

# However, the --noise parameter only decides how much noise each overlay has 
# that means, the higher the number, the more noise each frame would have 
# and they are random and all frames would have different noise
# therefore, there is no need for noise speed, as there is no need to move the noise in one way or another
# for different frames. 
parser.add_argument("--noise", type = float, default= 0, help = "the higher the absolute value of the number, the more noise each frame would have, and the noise are different.")

# structured_noise controls if we use the cifar-10 as background noise
parser.add_argument("--structured_noise", type = bool, default = False, help = "True: cifar-10 background noise would be added; False: not added")

# model specific parameters: 
parser.add_argument("--size", type = int, default = 1000, help = "total number of frames.")
parser.add_argument("--n_steps", type = int, default = 20, help = "Usually, we set the frame number to 20" )

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#wandb
os.environ["WANDB_MODE"] = "online"
wandb.init(
    project="ssl-vicreg-jepa",
    name="v-jepa-movingMNIST",
    mode="online", 
    config={
        "epochs": args.epochs,
        # this is just training batch size
        'size': args.size, 
        'batch_size': 1, 
        'n_frames': args.n_steps, 
        'action' : args.action, 
        'noise' : args.noise,
        'static_noise': args.static_noise,
        'static_noise_speed': args.static_noise_speed, 
        'structured_noise': args.structured_noise, 
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

# download cifar-10: 

# noise_cifar = torchvision.datasets.CIFAR10(root = "noise_cifar", 
# train = True, transform = None, target_transform = None, download = True)
# cifar_dataset = datasets.CIFAR10(root='noise_cifar', train=True, download=False)

## Create the moving dot datasets

class DatasetType(Enum):
    Single = auto()
    Multiple = auto()

'''
Level one: 
simplest way to create the dataset:
just a continuous dot moving dataset with action and/or noise
'''
dataset = ContinuousMotionDataset(size = args.size, batch_size = 1, n_steps=args.n_steps, 
                            noise = args.noise, static_noise = args.static_noise, 
                            static_noise_speed = args.static_noise_speed,  
                            structured_noise = args.structured_noise)
param_dict = {
    'size': args.size, 
    'batch_size': 1, 
    'n_frames': args.n_steps, 
    'action' : args.action, 
    'noise' : args.noise,
    'static_noise': args.static_noise,
    'static_noise_speed': args.static_noise_speed, 
    'structured_noise': args.structured_noise, 
}
print(param_dict)
print(len(dataset))

'''
Level 2: 
Three layer dataset overlayed over each other 
One dot not moving
One dot moving predictably
One dot moving unpredictably 
'''

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
# os.mkdir("./testing_frames/no_action_no_noise/")
# os.mkdir("./testing_frames/action_no_noise/")
# os.mkdir("./testing_frames/action_noise_2/")
# os.mkdir("./testing_frames/action_noise_0.1/")
# os.mkdir("./testing_frames/action_static_noise_0.1/")
# os.mkdir("./testing_frames/no_action_static_noise_0.3_speed_1/")
# os.mkdir("./testing_frames/action_static_noise_0.1_speed_1/")
def forward_pass(epoch, batch_num, encoder, predictor, batch, args):
    first_frame = batch[0][0][0][:][:]
    # print("first frame: ", first_frame)
    # print("first frame: ", frames[0][0][0][0][0])
    frames = batch.states.to(device)
    actions = batch.actions.to(device)
    # print("actions shape: ", actions.shape) # only 19 frames, because the last frame does not generate action 
    frames = frames.squeeze(1)
    actions = actions.squeeze(1)
    if args.action != None:
        predictor = predictor.to(device)
        seq_length = frames.shape[1]
        batch_size = frames.shape[0]

        actual_representations = []
        # action_representations = []
        total_cov_loss = 0.0
        total_var_loss = 0.0
        total_vc_loss = 0.0
        
        for t in range(seq_length):
            image_1 = frames[0, t].cpu().squeeze() 
            if epoch == 1 and batch_num == 1: 
                plt.figure(figsize=(6, 6))
                plt.imshow(image_1, cmap='gray')
                plt.axis('off')
                plt.title(f'epoch {epoch}, batch {batch_num}, frame {t}')
                
                plt.savefig(f'./testing_frames/action_static_noise_0.1_speed_1/frame_epoch{epoch}_batch{batch_num}_t{t}.png', 
                           bbox_inches='tight', dpi=150)
                plt.close()
                print(f"saved frame: frame_epoch{1}_batch{batch_num}_t{t}.png")
                print(f"frame shape: {image_1.shape}")
                print(f"frame value range: [{image_1.min():.3f}, {image_1.max():.3f}]")

            represenation = encoder(frames[:, t])
            # action_frames = even_frames[:, t]
            actual_representations.append(represenation)

            # flattened_action_frames = action_frames.flatten(start_dim = 1)
            # action_representations.append(flattened_action_frames)
            var_loss = variance_loss(represenation)
            cov_loss = covariance_loss(represenation)
            total_cov_loss += cov_loss
            total_var_loss += var_loss
            total_vc_loss += (cov_loss + var_loss)
        total_pred_loss = 0.0
        total_weighted_loss = 0.0
        predicted_representations = []
        current_representation = actual_representations[0]
        # current_action = action_representations[0]
        for t in range(seq_length-1):
            '''
            this is how to deal with the 
            dataset = ContinuousMotionDataset(size = 1000, batch_size = 1, n_steps=20)
            because the whole dataset is with unpredictoable movements (not the first layer of the
            three overlays). Therefore, we do not slice it, we just take the all action and send it 
            to predictor 
            '''
            # use 19 frames of action, beacuse we have 20 frames of videos
            current_action = actions[:,t]
            # if t == 0:
                # print("current_action: ", current_action)
                # print("current_action shape ", current_action.shape)
            # print("shapes: ", current_representation.shape, current_action.shape)
            # shapes:  torch.Size([32, 512]) torch.Size([32, 1, 2]) curr has been reshaped to 32, 512 after encoder
            # current_action is still 32, 1, 2, just once slice from each batch
            # we are going to send the one slice action to the predictor so that for each 19 frames, they will have 
            # one different action matched with them to go through the predictor and generate a St+1_pred
            # then this St+1_pred will be trained with St+1 with pred_loss 
            predicted_representation = predictor(current_representation, current_action)
            pred_loss = pred_loss_computations(predicted_representation, actual_representations[t+1])
            total_pred_loss += pred_loss
            current_representation = actual_representations[t+1]

    elif args.action == None: 
        predictor = predictor.to(device)
        print(frames.shape)
        seq_length = frames.shape[1]
        batch_size = frames.shape[0]

        actual_representations = []
        total_cov_loss = 0.0
        total_var_loss = 0.0
        total_vc_loss = 0.0
        for t in range(seq_length):
            image_1 = frames[0, t].cpu().squeeze() 
            if epoch == 1 and batch_num == 1: 
                plt.figure(figsize=(6, 6))
                plt.imshow(image_1, cmap='gray')
                plt.axis('off')
                plt.title(f'epoch {epoch}, batch {batch_num}, frame {t}')
                
                plt.savefig(f'./testing_frames/no_action_no_noise/frame_epoch{epoch}_batch{batch_num}_t{t}.png', 
                           bbox_inches='tight', dpi=150)
                plt.close()
                print(f"saved frame: frame_epoch{1}_batch{batch_num}_t{t}.png")
                print(f"frame shape: {image_1.shape}")
                print(f"frame value range: [{image_1.min():.3f}, {image_1.max():.3f}]")

            represenation = encoder(frames[:, t])
            # print(represenation.shape)
            actual_representations.append(represenation)
            var_loss = variance_loss(represenation)
            cov_loss = covariance_loss(represenation)
            total_cov_loss += cov_loss
            total_var_loss += var_loss
            total_vc_loss += (cov_loss + var_loss)

        total_pred_loss = 0.0
        predicted_representations = []
        current_representation = actual_representations[0]
        for t in range(seq_length - 1):
            predicted_representation = predictor(current_representation)
            pred_loss = pred_loss_computations(predicted_representation, actual_representations[t+1])
            total_pred_loss += pred_loss
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
        loss_dict = forward_pass(epoch, batch_num+1, encoder, predictor, batch, args)
     
        # TODO, should discuss
        total_weighted_loss = loss_dict["total_weighted_loss"]
        optimizer.zero_grad()
        total_weighted_loss.backward()
        optimizer.step()
        for key in epoch_loss_dict:
            if key != "epoch":
                epoch_loss_dict[key] += loss_dict[key]

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
            loss_dict = forward_pass(epoch, batch_num+1, encoder, predictor, batch, args)  
         
            total_weighted_loss = loss_dict["total_weighted_loss"]
            for key in epoch_loss_dict_val:
                if key != "epoch":
                    epoch_loss_dict_val[key] += loss_dict[key]

    epoch_logged_losses_val = {
        "epoch": epoch, 
        "total_var_loss": epoch_loss_dict_val['total_var_loss'].item()/total_batch_length,
        "total_cov_loss": epoch_loss_dict_val['total_cov_loss'].item()/total_batch_length,
        "total_vc_loss": epoch_loss_dict_val['total_vc_loss'].item()/total_batch_length,
        "total_pred_loss": epoch_loss_dict_val['total_pred_loss'].item()/total_batch_length,
        "total_weighted_loss": epoch_loss_dict_val['total_weighted_loss'].item()/total_batch_length
    }

    post_fix = "_val"
    new_epoch_logged_losses_val = {key + post_fix: value for key, value in epoch_logged_losses_val.items()}
    print(new_epoch_logged_losses_val)
    wandb.log({
        "total_var_loss_val": epoch_logged_losses_val['total_var_loss'], 
        "total_cov_loss_val": epoch_logged_losses_val['total_cov_loss'], 
        "total_vc_loss_val": epoch_logged_losses_val['total_vc_loss'], 
        "total_pred_loss_val": epoch_logged_losses_val['total_pred_loss'], 
        "total_weighted_loss_val": epoch_logged_losses_val['total_weighted_loss']
    })

    return new_epoch_logged_losses_val


dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)

print("after train loaders")
print(len(dataloader))

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [800, 200])
train_loader = DataLoader(train_dataset, args.batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True)
encoder = Encoder().to(device)


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

# 
if args.action != None:
    predictor = Predictor_with_action()
else:
    predictor = Predictor()
# args naming: 
if optimizer_name == "Adam":
    optim = Adam(list(encoder.parameters())+ list(predictor.parameters()), lr=learning_rate)
elif optimizer_name == "SGD":
    optim = torch.optim.SGD(list(encoder.parameters())+ list(predictor.parameters()), lr=learning_rate) 
elif optimizer_name == "RMSprop":
    optim = torch.optim.RMSprop(list(encoder.parameters())+ list(predictor.parameters()), lr=learning_rate)

print("no errors so far")

for epoch in range(1, num_epochs+1):
    train_loss_dict = train_one_epoch(epoch, encoder, predictor, train_loader, optim)
    val_loss_dict = val_one_epoch(epoch, encoder, predictor, val_loader)

wandb.finish()

