# a simple three layer Conv net encoder for SSL vicreg training 
# run with tran_vicreg.py 

import torch
import torch.nn as nn

class CNN_3Layer(nn.Module):
    def __init__(self, input_channels=1, output_dim=256):
        super(CNN_3Layer, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=8, stride=4, padding=2), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), 
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        # video 
        if x.dim() == 5:  
            B, C, T, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self.features(x)
            x = self.classifier(x)
            x = x.view(B, T, -1).mean(dim=1) 
        else:  
            x = self.features(x)
            x = self.classifier(x)
        
        return x