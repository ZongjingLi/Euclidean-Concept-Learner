import torch
from torch.autograd import Variable

v1 = Variable(torch.randn([1,32]))

print(v1)