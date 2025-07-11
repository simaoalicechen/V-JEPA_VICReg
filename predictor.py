# predictor use for train_jepa*.py
import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, representation_dim = 512, action_dim = 0):
        super().__init__()
        self.representation_dim = representation_dim
        self.action_dim = action_dim

        self.predictor = nn.Sequential(
                        nn.Linear(representation_dim + action_dim, 512), 
                        nn.ReLU(), 
                        nn.Linear(512, 512), 
                        nn.ReLU(), 
                        nn.Linear(512, representation_dim)
                    )

    def forward(self, _curr, action= None):
        x = _curr
        _next = self.predictor(_curr)
        return _next




