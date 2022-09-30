from .utils import *

import torch
import torch.nn as nn

from torch.autograd import Variable

from .config import *

import torch.distributions as dists

def point_wise_pdf(input_x,coord):
    """
    input_x: the grid of input that reprents the image
    coord  : the Nx2 shape tensor that represents locations in the grid
    """
    point_wise_normal = dists.Normal(coord,scale = opt.scale)
    
    return input_x

class ConceptModelSearch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x

class EuclidConceptModel(nn.Module):
    def __init__(self):
        super().__init__()
        """
        An euclid concept program is either a:
        point (x,y)  line(p1,p2) circle(p1,p2)
        or the composition of these things (constraint, and)
        """
        self.euclid_program = None

    def forward(self,x,format = "logp"):
        """
        logp          : return the log probability of image x is created by the model
        cross_entropy : return the cross entropy of the image x created by the model
        """
        return 0

