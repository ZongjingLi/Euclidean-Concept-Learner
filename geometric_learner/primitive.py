from turtle import circle
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from moic.data_structure import *
from moic.mklearn.nn.functional_net import *

import networkx as nx

def ptype(inputs):
    if inputs[0] == "c": return "circle"
    if inputs[0] == "l": return "line" 
    if inputs[0] == "p": return "point"

# geometric structure model. This is used to create the geometric concept graph
# and realize the concept and make sample of concepts. This is practically the decoder part of the 
# Geometric AutoEncoder model

dgc = "l1 = line(p1(), p2()),c1* = circle(p1(), p2()),c2* = circle(p2(), p1()),l2 = line(p1(), p3(c1, c2)),'l3 = line(p2(), p3())),"

def parse_geoclidean(program = dgc):
    left_bracket = 0
    for t in program:
        if t == "(" or t == "(":left_bracket += 1
        if t == ")" or t == ")":left_bracket -= 1
        if left_bracket == 0:pass
    return []


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
        if isinstance(concept_struct,str):concept_struct = parse_geoclidean(concept_struct)
        realized_graph  = nx.DiGraph()

        def parse_node(node):
            node_name = node.token
            
            # if the object is already in the graph, jsut return the name of the concept
            if node_name in realized_graph.nodes: return node_name
            visible = False if node_name[-1] == "*" else True
            realized_graph.add_node([node_name,visible])

            for child in node.children:
                realized_graph.add_edge(parse_node(child),node_name) # point from child to current node
    
            return node_name

        for program in concept_struct:parse_node(program)
        self.struct = realized_graph
    
        return realized_graph

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

if __name__ == "__main__":
    model = GeometricStructure()
    g = model.make_dag([toFuncNode("l1(p1(),p2())"),toFuncNode("l2(p2(),p3())"),toFuncNode("l3(p3(),p1())")])
    print(g.nodes)
    print(g.edges)
    nx.draw(g, with_labels=True, font_weight='bold')
    plt.show()

    g = model.make_dag([toFuncNode("l1(p1(),p2())"),toFuncNode("l2(p2(),p3())"),toFuncNode("l3(p3(),p1())")])
    print(g.nodes)
    print(g.edges)
    nx.draw(g, with_labels=True, font_weight='bold')
    plt.show()