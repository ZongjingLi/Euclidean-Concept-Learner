import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class GeometricConcept(nn.Module):
    def __init__(self):
        super().__init__()

class GeometricInvented(nn.Module):
    def __init__(self,program = "p()"):
        super().__init__()
        self.points  = []
        self.objects = []
    
    def realize(self):
        return