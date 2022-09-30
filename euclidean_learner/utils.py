import torch
from torch.autograd import Variable

v1 = Variable(torch.randn([1,32]),requires_grad = True)

def make_grid(resolution = (128,128)):
    # make the grid with the shape of [w,h]
    return resolution

def make_points(num = 1,distribution = "normal"):
    return [Variable(torch.randn(2),requires_grad = True) for _ in range(num)]

print(make_points(3))