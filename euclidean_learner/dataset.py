import torch
import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

class EuclidData(Dataset):
    def __init__(self,split = "train"):
        super().__init__()
    
    def __len__(self):return 0

    def __getitem__(self,index):return index
    