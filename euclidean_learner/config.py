import argparse

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu" 

parser = argparse.ArgumentParser()
parser.add_argument("--device",default = device)
parser.add_argument("--resolution",default = (64,64), type = tuple, help = "the default resolution for the image dataset")
parser.add_argument("--scale", default = 0.2, type = float, help = "the default scale for te line, point prob")
parser.add_argument("--line_scale",default = 0.6, type = float, help = "the line scale normal used for exist evaluation")
parser.add_argument("--lr", default = 1e-0, type = float, help = "the step length to adjust concept model into the real form")
parser.add_argument("--info_dim",default = 132, type = int, help = "the message passing dim for the concept structure realization")

opt = parser.parse_args(args = [])