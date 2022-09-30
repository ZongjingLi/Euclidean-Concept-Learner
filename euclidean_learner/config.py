import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--resolution",default = (128,128), type = tuple, help = "the default resolution for the image dataset")
parser.add_argument("--scale", default = 0.2, type = float, help = "the default scale for te line, point prob")

opt = parser.parse_args(args = [])