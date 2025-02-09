from re import S
from turtle import down
import torch
import torch.nn as nn
import torch.distributions as dists

import numpy as np

import math

import matplotlib.pyplot as plt
from moic.data_structure import *
from moic.mklearn.nn.functional_net import *

import networkx as nx

try:from .config import *
except:from config import *

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

def union_pdf(components):
    first_comp = components[0]
    for comp in components: first_comp = torch.max(comp,first_comp).values
    return first_comp

def intersect_pdf(components):
    first_comp = components[0]
    for comp in components: first_comp = torch.min(comp,first_comp).values
    return first_comp

def segment(start,end,segments):
    outputs = [ start + (end-start) * i/segments for i in range(segments)]
    return torch.stack(outputs)

class PointProp(nn.Module):
    def __init__(self,opt):
        super().__init__()
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

class MessageProp(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.update_map   = nn.Linear(opt.geometric_latent_dim,opt.geometric_latent_dim)
        self.message_map  = nn.Linear(opt.geometric_latent_dim,opt.geometric_latent_dim)
        self.joint_update = FCBlock(132,2,2 *  opt.geometric_latent_dim,opt.geometric_latent_dim)
        self.opt = opt

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

def make_grid(resolution = (64,64)):
    # make the grid with the shape of [w,h]
    x = torch.linspace(0,resolution[0],resolution[0])
    y = torch.linspace(0,resolution[1],resolution[1])
    grid = torch.meshgrid(x,y)
    return torch.stack(grid)

def sample_point(pdf):
    W,H      = pdf.shape
    norm_pdf = pdf.flatten().numpy()
    norm_pdf = norm_pdf/np.sum(norm_pdf,-1)
    grid     = make_grid([W,H]).permute([1,2,0]).flatten(start_dim = 0,end_dim = 1).numpy()

    location = np.random.choice(range(norm_pdf.shape[0]),p = norm_pdf)

    return np.int8(grid[location])

class GeometricStructure(nn.Module):
    def __init__(self,opt = model_opt):
        super().__init__()
        # structure stored in the realization
        self.realized = False
        self.struct = None
        self.grid   = None
        self.visible = []

        # TODO: implement another version of the line propagator so the input is invariant
        self.line_propagator = FCBlock(132,2,opt.geometric_latent_dim * 2, opt.geometric_latent_dim)
        self.circle_propagator  = FCBlock(132,2,opt.geometric_latent_dim * 2, opt.geometric_latent_dim)
        self.point_propagator   = PointProp(opt)
        # this line above is the upward proppagation
        self.message_propagator = MessageProp(opt) # this is the message proppagator used in the downward proppagation

        # TODO: make a better version of the downward propagation

        # graph signal storage
        self.upward_memory_storage   = None
        self.downward_memory_storage = None

        # decode the semantics of the signal vector
        self.signal_decoder = RenderField(model_opt)
        self.opt = opt

        self.clear()

    def clear(self):
        self.grid     = make_grid(self.opt.resolution).permute([1,2,0]).to(self.opt.device)
        self.realized = False # clear the state of dag, and the realization
        self.struct   = None  # clear the state of concept struct   
        self.upward_memory_storage   = None
        self.downward_memory_storage = None

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
                left_component   = quest_down(connect_to[0]);right_component  = quest_down(connect_to[1])
                update_component = self.circle_propagator(torch.cat([left_component,right_component],-1))
            if primitive_type == "line":
                assert len(connect_to) == 2,print("the line is connected to {} parameters (2 expected).".format(len(connect_to)))
                start_component  = quest_down(connect_to[0]);end_component    = quest_down(connect_to[1])
                update_component = self.line_propagator(torch.cat([start_component,end_component],-1))
            if primitive_type == "point":
                point_prop_inputs = []
                for component in connect_to:
                    if component == "<V>": # the input prior is in the domain of <V>
                        pass# TODO:point_prop_inputs.append(signal)
                    else:point_prop_inputs.append(quest_down(component)) # the input prior is the intersection of some component
                update_component = self.point_propagator(signal,point_prop_inputs)
            if node == "<V>":return
        
            upward_memory_storage[node] = update_component 
            return update_component
        
        for node in self.struct.nodes:quest_down(node)
        # update the memory unit after the propagation
        self.upward_memory_storage   = upward_memory_storage

        # 2. start the downward propagation. (maybe not)
        downward_memory_storage   = {}
        def quest_up(node):
            if node == "<V>":return
            if node in downward_memory_storage:return downward_memory_storage[node]# if node already calculated, nothing happens
            connect_to     =  find_connection(node,self.struct,loc = 0) # find all the nodes that connected to the current node

            input_neighbors = [quest_up(p_node) for p_node in connect_to]
            current_node_feature = self.upward_memory_storage[node] # this is the feature a point store currently (circle,point,line aware)
            update_component = self.message_propagator(current_node_feature,input_neighbors) # this is the update component feature
        
            downward_memory_storage[node] = update_component 
            return update_component
        for node in self.struct: quest_up(node)

        # update the memory unit of the propagation
        self.downward_memory_storage = downward_memory_storage
        return 

    def sample(self,log = False):
        assert self.struct is not None,print("the dag struct is None") 
        assert self.realized,print("This concept dag is not realized yet")
        
        calculated_pdf = {}
        output_grid = torch.zeros(self.opt.resolution + (1,)) # every time a pdf is composed with the current one
        def Pr(node):
            if node in calculated_pdf:return calculated_pdf[node] # just take the memory if this node is calculated
            node_type  = ptype(node)
            connect_to = find_connection(self.struct,loc = 1) # find all the points that this point connected by

            if node_type == "line":
                assert len(connect_to) == 2,print("the line is connected to {} parameters (2 expected).".format(len(connect_to)))
                point1_pdf   = Pr(connect_to[0]);point2_pdf = Pr(connect_to[1])
                point1_coord = sample_point(point1_pdf);point2_coord = sample_point(point2_pdf)
                segments = math.abs(point1_coord[0] - point2_coord[0])

                grid_expand = self.grid.flatten(start_dim = 0, end_dim = 1).unsqueeze(0).repeat([self.segments,1,1])
                self.line = segment(point1_coord,point2_coord,segments)
                diff = grid_expand - self.line.unsqueeze(1).repeat([1,self.resolution[0] * self.resolution[1],1])

                leng_diff = torch.norm(diff,2,dim = -1)
                min_diff = torch.min(leng_diff,0).values

                line_norm = dists.Normal(0,self.opt.line_scale)
                logpdf = line_norm.log_prob(min_diff)

                logpdf = logpdf.view(self.opt.resolution)
                update_pdf = logpdf.exp() if not log else logpdf


            if node_type == "circle":
                assert len(connect_to) == 2,print("the line is connected to {} parameters (2 expected).".format(len(connect_to)))
                point1_pdf = Pr(connect_to[0]);point2_pdf = Pr(connect_to[1])
                update_pdf = 0

            if node_type == "point":
                # calculate the render field made 
                attention_field = self.signal_decoder(self.upward_memory_storage[node])
                if log:attention_field = attention_field.log()
                
                # the connections to this point will be constraints
                if len(connect_to) == 0:update_pdf = attention_field
                else:
                    constraint = [Pr(obj) for obj in connect_to]
                    update_pdf = intersect_pdf(constraint)

            if node in self.visible: grid = union_pdf(update_pdf,grid) # add the pdf onto the grid
        
        # for node in self.struct.nodes:Pr(node)
        return output_grid

# this is a neural render field defined on 2D grids. Input a semantics vector, it will output a attention 
# map that represents the attention realm on the grid domain

class PointDecoder(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.raw_decoder = FCBlock(132,4,in_dim,2)
        self.alpha = 7;self.beta = 64
    def forward(self,x):return torch.sigmoid( self.alpha *  self.raw_decoder(x) ) * self.beta

class RenderField(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.render_field = FCBlock(132,3,2+opt.geometric_latent_dim,1)
        self.gamma = 7
    
    def forward(self,grid,infos):
        # input grid: BxWxHx2 input info: BxS
        B,W,H,_ = grid.shape
        expand_info = infos.unsqueeze(1).unsqueeze(1)
        expand_info = expand_info.repeat([1,W,H,1])
        return torch.sigmoid(self.gamma * self.render_field(torch.cat(grid,expand_info),-1))

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