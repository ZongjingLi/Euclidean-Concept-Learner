import argparse

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu" 

parser = argparse.ArgumentParser()
parser.add_argument("--device",default = device)
parser.add_argument("--resolution",default = (128,128), type = tuple, help = "the default resolution for the image dataset")
parser.add_argument("--scale", default = 1., type = float, help = "the default scale for te line, point prob")
parser.add_argument("--line_scale",default = 0.5, type = float, help = "the line scale normal used for exist evaluation")

opt = parser.parse_args(args = [])