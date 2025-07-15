# predictor use for train_jepa_movingDots.py
import torch
import torch.nn as nn

# dot with incomplete action 
# shapes:  torch.Size([32, 512]) torch.Size([32, 19, 1, 2]) rpre and action 
class Predictor_with_action(nn.Module):
    def __init__(self):
        #  representation_dim = 512, action_dim = 0):
        super().__init__()
        self.rep_branch = nn.Sequential(
            nn.Linear(512, 512), 
            nn.ReLU(), 
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.action_branch = nn.Sequential(
            nn.Linear(19*1*2, 256), 
            nn.ReLU(), 
            nn.Linear(256, 512), 
            nn.ReLU(), 
            nn.Linear(512, 128), 
        )
        self.combined_layer = nn.Linear(128 + 128, 64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, curr_represenation, action):
        rpre_output = self.rep_branch(curr_represenation)
        # action_output = self.action_branch(action)
        action_flat = action.view(action.size(0), -1)
        action_output = self.action_branch(action_flat)
        combined_output = torch.cat((rpre_output, action_output), dim=1)
        output = self.output_layer(self.combined_layer(combined_output))
        return output

