import torch
import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--device",default = device, type = str, help = "the default running device will be on cuda is available")
parser.add_argument("--point_scale",default = 0.3, type = float, help = "the variance of the point like distribution")
parser.add_argument("--resolution", default = (64,64), type = tuple, help = "the global resolution of the input concept image")
parser.add_argument("--geometric_latent_dim", default = 32, type = int, help = "the default latent dim of inside the geometric structure")
parser.add_argument("--encoder_latent_dim", default = 64, type = int, help = "the latent dim of the encoder module")

parser.add_argument("--scale", default = 0.2, type = float, help = "the default scale for te line, point prob")
parser.add_argument("--line_scale",default = 0.6, type = float, help = "the line scale normal used for exist evaluation")

model_opt= parser.parse_args(args = [])