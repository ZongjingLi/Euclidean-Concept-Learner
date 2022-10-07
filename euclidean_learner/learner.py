from encoder import * # import all the encoder models
from model   import * # import all the concept models

from config import * # load the geometric model config
"""
This Geometric Learner is practically a Auto-Encoder given a fixed concept structure
    (encoder): geometric encoder that encoder the global information
    (decoder): the concept structure created by the 
"""
class GeometricLearner(nn.Module):
    def __init__(self,model_opt):
        super().__init__()
        self.encoder = GeometricEncoder(3,64)
        self.concept = None

    def forward(self,x):
        feature_map = self.encoder(x)
        print(feature_map.shape)
        return feature_map
