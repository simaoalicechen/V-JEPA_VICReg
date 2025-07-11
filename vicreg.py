# from https://arxiv.org/pdf/2105.04906
# the function definitions used for train_jepa*.py are in train_jepa*.py

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def vicreg_loss(z1, z2, inv_coeff = 25.0, var_coeff = 15.0, cov_coeff = 1.0):
    # batch_size, embedding_dim = z1.shape

    # how similar two embeddings are, use distance based loss 
    # minimize 
    invariance_loss = torch.nn.functional.mse_loss(z1, z2)

    # varaince loss, how different two embeddings are
    # make sure they are still diverse 
    # maximize 
    def variance_loss(x, gamma):
        x = x - x.mean(dim=0)
        std = x.std(dim=0)
        var_loss = F.relu(gamma - std).mean()
        return var_loss 

    # how one embedding correlate with the other
    # minimize it
    def covariance_loss(x):
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (x.shape[0] - 1)
        cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / x.shape[1]
        return cov_loss

    z1_variance_loss = variance_loss(z1, 1e-04)
    z2_varaince_loss = variance_loss(z2, 1e-04)
    z1_covariance_loss = covariance_loss(z1)
    z2_covariance_loss = covariance_loss(z2)

    variance_loss = (z1_variance_loss + z2_varaince_loss)/2
    covariance_loss = (z1_covariance_loss + z2_covariance_loss)/2
    loss = inv_coeff*invariance_loss + var_coeff*variance_loss + cov_coeff*covariance_loss
    return loss, variance_loss, covariance_loss, invariance_loss 

# def vc_loss(z1, z2)


def vc_loss(z1, z2, var_coeff = 15.0, cov_coeff = 1.0):

    # varaince loss, how different two embeddings are
    # make sure they are still diverse 
    # maximize 
    def variance_loss(x, gamma):
        x = x - x.mean(dim=0)
        std = x.std(dim=0)
        var_loss = F.relu(gamma - std).mean()
        return var_loss 

    # how one embedding correlate with the other
    # minimize it
    def covariance_loss(x):
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (x.shape[0] - 1)
        cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / x.shape[1]
        return cov_loss

    z1_variance_loss = variance_loss(z1, 1e-04)
    z2_varaince_loss = variance_loss(z2, 1e-04)
    z1_covariance_loss = covariance_loss(z1)
    z2_covariance_loss = covariance_loss(z2)

    variance_loss = (z1_variance_loss + z2_varaince_loss)/2
    covariance_loss = (z1_covariance_loss + z2_covariance_loss)/2
    loss = var_coeff*variance_loss + cov_coeff*covariance_loss
    return loss, variance_loss, covariance_loss


# def inv_loss(z1, z2, inv_coeff = 25.0):
#     # batch_size, embedding_dim = z1.shape

#     # how similar two embeddings are, use distance based loss 
#     # minimize 
#     invariance_loss = torch.nn.functional.mse_loss(z1, z2)

#     invariance_loss = inv_coeff*invariance_loss
#     return invariance_loss

