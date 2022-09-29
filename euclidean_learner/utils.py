import torch
from torch.autograd import Variable

v1 = Variable(torch.randn([1,32]),requires_grad = True)

def make_grid(resolution = (128,128)):
    # make the grid with the shape of [w,h]
    return resolution