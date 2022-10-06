import torch
import torch.nn as nn

# create a visual encoder that stores global informaiton

class GeometricEncoder(nn.Module):
    def __init__(self,model_opt):
        super().__init__()

    def forward(self,x):
        return x