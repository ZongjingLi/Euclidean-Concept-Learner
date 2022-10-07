from encoder import * # import all the encoder models
from model   import * # import all the concept models

from config import * # load the geometric model config

class GeometricLearner(nn.Module):
    def __init__(self,model_opt):
        super().__init__()
        self.encoder = GeometricEncoder()
        self.concept = None

    def forward(self,x):
        return x
