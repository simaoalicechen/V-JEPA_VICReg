# predictor use for train_jepa*.py
import torch
import torch.nn as nn

# process the inputs (actual representations and action representations seperately with 
# different branches, aka sub-networks, with one predictor)
# can change the dimension to suit with other types of actions if necessary 
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
            nn.Linear(4096, 1024), 
            nn.ReLU(), 
            nn.Linear(1024, 1024), 
            nn.ReLU(), 
            nn.Linear(1024, 128), 
        )
        self.combined_layer = nn.Linear(128 + 128, 64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, curr_represenation, action):
        rpre_output = self.rep_branch(curr_represenation)
        action_output = self.action_branch(action)
        combined_output = torch.cat((rpre_output, action_output), dim=1)
        output = self.output_layer(self.combined_layer(combined_output))
        return output




