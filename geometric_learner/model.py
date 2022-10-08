from .config     import *
from .encoder    import *
from .dataloader import *
from .primitive  import *

import torch
import torch.nn as nn


# This is used to create a geometric auto-encoder

class GeometricAutoEncoder(nn.Module):
    def __init__(self,model_opt):
        super().__init__()
        self.encoder = GeometricEncoder(3,132)
        self.asearch = None
        self.decoder = GeometricStructure()
        self.lateral = FCBlock(132,3,64 * 64 * 132,model_opt.encoder_latent_dim)

    def find_concept_struct(self,image):
        return 0

    def forward(self,image,concept_struct = None):
        """
        inoput concept struct should be a list of func-node programs
        """
        if isinstance(concept_struct,str):print("concept struct is an instance")
        if concept_struct is None:concept_struct = self.find_concept_struct(image)
        # encode input features
        encoder_features = self.encoder(image) # encoder the global prior feature
        encoder_features = self.lateral(encoder_features.flatten(start_dim = 1))

        # construct the output using the concept structure
        self.decoder.clear() # clear the previous storage of dag and embeddings.
        self.decoder.make_dag(concept_struct)  # build the dag graph of the concept structure.
        self.decoder.realize(encoder_features) # realize the semantics information and store in each.

        # decode the output image
        sample = self.decoder.sample()

        return sample

    def namomo(self):return "namomo"


if __name__ == "__main__":
    pass