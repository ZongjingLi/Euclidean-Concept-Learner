from encoder import * # import all the encoder models
from model   import * # import all the concept models

from config import * # load the geometric model config
"""
This Geometric Learner is practically a Auto-Encoder given a fixed concept structure
    (encoder): geometric encoder that encoder the global information
    (decoder): the concept structure that use the latent-z as prior to realize a concept
    
    problems: how is the concept model realized based on the latent-z
              what is the structure of concept relation model
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
