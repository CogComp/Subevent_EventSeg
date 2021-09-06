############################################
# Rectifier Network                        #
############################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class RectifierNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RectifierNetwork, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.sigmoid = torch.nn.Sigmoid()
        #torch.nn.init.xavier_uniform(self.fc1.weight)

    def forward(self, inp):
        out_intermediate = F.relu(self.fc1(inp))
        output = self.sigmoid(1 - torch.sum(out_intermediate, 1))
        return output