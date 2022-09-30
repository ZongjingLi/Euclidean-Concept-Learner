from .utils import *

import torch
import torch.nn as nn

from torch.autograd import Variable

from .config import *

class ConceptModelSearch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x