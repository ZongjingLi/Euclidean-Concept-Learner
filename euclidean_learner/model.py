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
        self.resolution = resolution
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

    def exist(self,x,log = True):
        pdf = self.pdf(log).unsqueeze(-1)

        return torch.sum(pdf * x,-1)

class EuclidLineModel(EuclidConceptModel):
    def __init__(self,point1 = None,point2 = None):
        super().__init__()

        self.segments = 100
        
        if point1 is None:
            self.point1 = EuclidPointModel(None) # if there is no given point

        elif isinstance(point1,EuclidPointModel):
            self.point1 = point1 # if the parameter is a give point
        else:
            self.point1 = EuclidPointModel(point1) # there is fixed given point 
        
        if point2 is None:
            self.point2 = EuclidPointModel(None) # if there is no given point

        elif isinstance(point2,EuclidPointModel):
            self.point2 = point2 # if the parameter is a give point
        else:
            self.point2 = EuclidPointModel(point2) # there is fixed given point
        self.line = segment(self.point1.coord,self.point2.coord,self.segments)

    def pdf(self,log = True):
        grid_expand = self.grid.flatten(start_dim = 0, end_dim = 1).unsqueeze(0).repeat([self.segments,1,1])
        diff = grid_expand - self.line.unsqueeze(1).repeat([1,self.resolution[0] * self.resolution[1],1])

        leng_diff = torch.norm(diff,2,dim = -1)
        min_diff = torch.min(leng_diff,0).values

        line_norm = dists.Normal(0,opt.line_scale)
        logpdf = line_norm.log_prob(min_diff)

        logpdf = logpdf.view(opt.resolution)
        if log:return logpdf
        return  logpdf.exp()

    def exist(self,x,log = True):
        pdf = self.pdf(log).unsqueeze(-1)
        return torch.sum(pdf * x,-1)


def adjust_model_to_observation(model,x,n_epochs = 50,visualize = True):
    optim = torch.optim.Adam(model.parameters(), lr = opt.lr)
    for epoch in range(n_epochs):
        optim.zero_grad()
        logpdf = model.exist(x)
        loss = 0 - torch.sum(logpdf)
        loss.backward()
        optim.step()
        logpdf = model.pdf(False)
        if visualize:
            print(epoch,loss)
            plt.imshow(logpdf.detach());plt.pause(0.01);plt.cla()