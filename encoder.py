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
            nn.Linear(256*1*1, 512), 
            # 28*28 = 256
            nn.ReLU(), 
            nn.Linear(512, representation_dim)
        )


    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        return x

# key: only make sure the output dimensions 512 and 512 match the ones in predictor's input dimension, 512*512
# leave the rest to gradually go down and up. 