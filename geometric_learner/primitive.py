from turtle import circle
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from moic.mklearn.nn.functional_net import *

import networkx as nx

def ptype(inputs):
    if inputs[0] == "c": return "circle"
    if inputs[0] == "l": return "line" 
    if inputs[0] == "p": return "point"

# geometric structure model. This is used to create the geometric concept graph
# and realize the concept and make sample of concepts. This is practically the decoder part of the 
# Geometric AutoEncoder model

class GeometricStructure(nn.Module):
    def __init__(self,program = "p1()"):
        super().__init__()
        self.points  = []
        self.objects = []
        self.realized = False
        self.struct = None

    def make_dag(self,concept_struct):
        """
        input:  the concept struct is a list of func nodes
        output: make self.struct as a list of nodes and edges
        """
        return 0

    def realize(self):
        return

    def sample(self):
        assert self.struct is not None,print("the dag struct is None")
        assert self.realized,print("This concept dag is not realized yet")


# this is a neural render field defined on 2D grids. Input a semantics vector, it will output a attention 
# map that represents the attention realm on the grid domain

class RenderField(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.render_field = FCBlock(132,3,2+opt.info_dim,1)
    
    def forward(self,grid,infos):
        # input grid: BxWxHx2 input info: BxS
        B,W,H,_ = grid.shape
        expand_info = infos.unsqueeze(1).unsqueeze(1)
        expand_info = expand_info.repeat([1,W,H,1])
        return self.render_field(torch.cat(grid,expand_info),-1)