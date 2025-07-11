# encoder use for train_jepa*.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_channels = 1, representation_dim = 512):
        super().__init__()
        self.representation_dim = representation_dim

        self.conv = nn.Sequential(
                        nn.Conv2d(1, 32, 4, 2, 1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 4, 2, 1),
                        nn.ReLU(),
                        nn.Conv2d(64, 128, 4, 2, 1), 
                        nn.ReLU(), 
                        nn.Conv2d(128, 256, 4, 2, 1), 
                        nn.ReLU(),
                    )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, representation_dim), 
            nn.ReLU(), 
            nn.Linear(representation_dim, representation_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
