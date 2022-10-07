from turtle import circle
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

def ptype(inputs):
    if inputs[0] == "c": return "circle"
    if inputs[0] == "l": return "line" 
    if inputs[0] == "p": return "point"

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