from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def vicreg_loss(z1, z2, inv_coeff = 25.0, var_coeff = 15.0, cov_coeff = 1.0):
    # batch_size, embedding_dim = z1.shape

    # how similar two embeddings are, use distance based loss 
    invariance_loss = torch.nn.functional.mse_loss(z1, z2)

    # varaince loss, how different two embeddings are
    def variance_loss(x, gamma):
        x = x - x.mean(dim=0)
        std = x.std(dim=0)
        var_loss = F.relu(gamma - std).mean()
        return var_loss 

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


# class VICRegLoss(nn.Module):
#     def __init__(
#         self,
#         inv_coeff: float = 25.0,
#         var_coeff: float = 15.0,
#         cov_coeff: float = 1.0,
#         gamma: float = 1.0,
#     ):
#         super().__init__()
#         self.inv_coeff = inv_coeff
#         self.var_coeff = var_coeff
#         self.cov_coeff = cov_coeff
#         self.gamma = gamma

    # def vicreg_loss(z1, z2, inv_coeff = self.)

    # def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
    #     """Computes the VICReg loss.

    #     ---
    #     Args:
    #         x: Features map.
    #             Shape of [batch_size, representation_size].
    #         y: Features map.
    #             Shape of [batch_size, representation_size].

    #     ---
    #     Returns:
    #         The VICReg loss.
    #             Dictionary where values are of shape of [1,].
    #     """
    #     metrics = dict()
    #     metrics["inv-loss"] = self.inv_coeff * self.representation_loss(x, y)
    #     metrics["var-loss"] = (
    #         self.var_coeff
    #         * (self.variance_loss(x, self.gamma) + self.variance_loss(y, self.gamma))
    #         / 2
    #     )
    #     metrics["cov-loss"] = (
    #         self.cov_coeff * (self.covariance_loss(x) + self.covariance_loss(y)) / 2
    #     )
    #     metrics["loss"] = sum(metrics.values())
    #     return metrics

    # @staticmethod
    # def representation_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    #     """Computes the representation loss.
    #     Force the representations of the same object to be similar.

    #     ---
    #     Args:
    #         x: Features map.
    #             Shape of [batch_size, representation_size].
    #         y: Features map.
    #             Shape of [batch_size, representation_size].

    #     ---
    #     Returns:
    #         The representation loss.
    #             Shape of [1,].
    #     """
    #     return F.mse_loss(x, y)

    # @staticmethod
    # def variance_loss(x: torch.Tensor, gamma: float) -> torch.Tensor:
    #     """Computes the variance loss.
    #     Push the representations across the batch
    #     to be different between each other.
    #     Avoid the model to collapse to a single point.

    #     The gamma parameter is used as a threshold so that
    #     the model is no longer penalized if its std is above
    #     that threshold.

    #     ---
    #     Args:
    #         x: Features map.
    #             Shape of [batch_size, representation_size].

    #     ---
    #     Returns:
    #         The variance loss.
    #             Shape of [1,].
    #     """
    #     x = x - x.mean(dim=0)
    #     std = x.std(dim=0)
    #     var_loss = F.relu(gamma - std).mean()
    #     return var_loss

    # @staticmethod
    # def covariance_loss(x: torch.Tensor) -> torch.Tensor:
    #     """Computes the covariance loss.
    #     Decorrelates the embeddings' dimensions, which pushes
    #     the model to capture more information per dimension.

    #     ---
    #     Args:
    #         x: Features map.
    #             Shape of [batch_size, representation_size].

    #     ---
    #     Returns:
    #         The covariance loss.
    #             Shape of [1,].
    #     """
    #     x = x - x.mean(dim=0)
    #     cov = (x.T @ x) / (x.shape[0] - 1)
    #     cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / x.shape[1]
    #     return cov_loss