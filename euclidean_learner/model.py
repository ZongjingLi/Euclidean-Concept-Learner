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
    def __init__(self,resolution = (128,128)):
        super().__init__()
        """
        An euclid concept program is either a:
        point (x,y)  line(p1,p2) circle(p1,p2)
        or the composition of these things (constraint, and)
        """
        self.euclid_program = None
        self.grid = make_grid(resolution).permute([1,2,0]).to(opt.device)

    def forward(self,x,format = "logp"):
        """
        logp          : return the log probability of image x is created by the model
        cross_entropy : return the cross entropy of the image x created by the model
        """
        return 0

class EuclidPointModel(EuclidConceptModel):
    def __init__(self,coord = None):
        super().__init__()
        self.coord = nn.Parameter(coord) if coord is not None else nn.Parameter(torch.randn([1,2]))

    def pdf(self,log = True):
        point_wise_normal = dists.Normal(self.coord,opt.scale)
        if log:return torch.sum(point_wise_normal.log_prob(self.grid),-1)
        return  torch.sum(point_wise_normal.log_prob(self.grid),-1).exp()