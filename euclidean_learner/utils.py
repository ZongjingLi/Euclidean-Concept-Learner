import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

def make_grid(resolution = (128,128)):
    # make the grid with the shape of [w,h]
    x = torch.linspace(0,resolution[0],resolution[0])
    y = torch.linspace(0,resolution[1],resolution[1])
    grid = torch.meshgrid(x,y)
    return torch.stack(grid)

def make_points(num = 1,distribution = "normal"):
    return [Variable(torch.randn(2),requires_grad = True) for _ in range(num)]


def gain_distribution(self,pdf):
    return 0
    
def segment(start,end,segments):
    outputs = [ start + (end-start) * i for i in range(segments)]
    return torch.stack(outputs)