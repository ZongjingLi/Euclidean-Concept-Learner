import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from moic.data_structure import *
from moic.mklearn.nn.functional_net import *

import networkx as nx
from .config import *
#from config import *

def ptype(inputs):
    if inputs[0] == "c": return "circle"
    if inputs[0] == "l": return "line" 
    if inputs[0] == "p": return "point"

# geometric structure model. This is used to create the geometric concept graph
# and realize the concept and make sample of concepts. This is practically the decoder part of the 
# Geometric AutoEncoder model

dgc = ["l1 = line(p1(), p2())","c1* = circle(p1(), p2())","c2* = circle(p2(), p1())","l2 = line(p1(), p3(c1, c2))","l3 = line(p2(), p3()))"]

def parse_geoclidean(programs = dgc):
    outputs = []
    for program in programs:
        left,right = program.split("=")
        left = left.replace(" ","");right = right.replace(" ","")
        func_node_form = toFuncNode(right)
        func_node_form.token = left
        outputs.append(func_node_form)
    return outputs

class PointProp(nn.Module):
    def __init__(self,opt):
        super().__init__()
        magic_number = 132
        self.opt = opt
        self.update_map   = nn.Linear(opt.geometric_latent_dim,opt.geometric_latent_dim)
        self.message_map  = nn.Linear(opt.geometric_latent_dim,opt.geometric_latent_dim)
        self.joint_update = FCBlock(132,2,opt.encoder_latent_dim + opt.geometric_latent_dim,opt.geometric_latent_dim)

    def forward(self,signal,components):
        if not components: 
            return self.joint_update(torch.cat([signal,torch.zeros([1,self.opt.geometric_latent_dim])] ,-1))
        right_inters = 0
        for comp in components:right_inters += self.message_map(comp)

        right_inters = self.update_map(right_inters)
        return self.joint_update(torch.cat([signal,right_inters],-1))

def find_connection(node,graph,loc = 0):
    outputs = []
    for edge in graph.edges:
        if edge[loc] == node:outputs.append(edge[int(not loc)])
    return outputs

class GeometricStructure(nn.Module):
    def __init__(self,opt = model_opt):
        super().__init__()
        # structure stored in the realization
        self.realized = False
        self.struct = None
        self.visible = []

        # TODO: implement another version of the line propagator so the input is invariant
        self.line_propagator = FCBlock(132,2,opt.geometric_latent_dim * 2, opt.geometric_latent_dim)
        self.circle_propagator  = FCBlock(132,2,opt.geometric_latent_dim * 2, opt.geometric_latent_dim)
        self.point_propagator = PointProp(opt)

    def clear(self):
        self.realized = False # clear the state of dag, and the realization
        self.struct   = None  # clear the state of concept struct

    def make_dag(self,concept_struct):
        """
        input:  the concept struct is a list of func nodes
        output: make self.struct as a list of nodes and edges
        """
        if isinstance(concept_struct[0],str):concept_struct = parse_geoclidean(concept_struct)
        realized_graph  = nx.DiGraph()
        self.visible = []

        def parse_node(node):
            node_name = node.token
            
            # if the object is already in the graph, jsut return the name of the concept
            if node_name in realized_graph.nodes: return node_name
            if node_name == "":# if this place is a void location.
                node_name = "<V>";visible = False
            elif node_name[-1] == "*":
                node_name = node_name.replace("*","");visible =False
            else:visible = True
            realized_graph.add_node(node_name)
            for child in node.children:
                if visible:self.visible.append(node_name)
                realized_graph.add_edge(parse_node(child),node_name) # point from child to current node
    
            return node_name

        for program in concept_struct:parse_node(program)
        self.struct = realized_graph
        self.realized = True
    
        return realized_graph

    def realize(self,signal):
        # given every node a vector representation
        # 1. start the upward propagation
        upward_memory_storage   = {}
        def quest_down(node):
            if node in upward_memory_storage:return upward_memory_storage[node]# is it is calculated, nothing happens
            primitive_type =  ptype(node)
            connect_to     =  find_connection(node,self.struct,loc = 1)
            if primitive_type == "circle": # use the circle propagator to calculate mlpc(cat([ec1,ec2]))
                assert len(connect_to) == 2,print("the circle is connected to {} parameters (2 expected).".format(len(connect_to)))
                left_component   = quest_down(connect_to[0])
                right_component  = quest_down(connect_to[1])
                update_component = self.circle_propagator(torch.cat([left_component,right_component],-1))
            if primitive_type == "line":
                assert len(connect_to) == 2,print("the line is connected to {} parameters (2 expected).".format(len(connect_to)))
                start_component  = quest_down(connect_to[0])
                end_component    = quest_down(connect_to[1])
                update_component = self.line_propagator(torch.cat([start_component,end_component],-1))
            if primitive_type == "point":
                point_prop_inputs = []
                for component in connect_to:
                    if component == "<V>": # the input prior is in the domain of <V>
                        pass#point_prop_inputs.append(signal)
                    else: # the input prior is the intersection of some component
                        point_prop_inputs.append(quest_down(component))
                update_component = self.point_propagator(signal,point_prop_inputs)
            if node == "<V>":return
        
            upward_memory_storage[node] = update_component 
            return update_component
        
        for node in self.struct.nodes:quest_down(node)

        # 2. start the downward propagation. (maybe not)
        # TODO: downward propagation of the dag 
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
    g = model.make_dag(["l1 = line(p1(),p2())","l2 = line(p2(),p3())","l3 = line(p3(),p1())"])

    model.realize(torch.randn([1,model_opt.encoder_latent_dim]))

    nx.draw(g, with_labels=True, font_weight='bold')
    plt.show()

    print("phase 2")    
    g = model.make_dag(dgc)

    model.realize(torch.randn([1,model_opt.encoder_latent_dim]))

    nx.draw(g, with_labels=True, font_weight='bold')
    plt.show()

    print(g.edges)