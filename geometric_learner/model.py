from config import *
from encoder import *

import torch
import torch.nn as nn

# This is used to create a geometric auto-encoder

class GeometricAutoEncoder(nn.Module):
    def __init__(self,model_opt):
        super().__init__()
        self.encoder = GeometricEncoder(3,132)
        self.asearch = None
        self.structure = None

    def find_concept_struct(self,image):
        return 0

    def forward(self,image,concept_struct = None):
        """
        inoput concept struct should be a list of func-node programs
        """
        if isinstance(concept_struct,str):print("concept struct is an instance")
        if concept_struct is None:concept_struct = self.find_concept_struct(image)
        encoder_features = self.encoder(image) # encoder the global prior feature

        self.structure.make_dag(concept_struct)
        self.structure.realize(encoder_features) # use the 

        sample = self.structure.sample()

        return sample

    def namomo(self):return "namomo"

