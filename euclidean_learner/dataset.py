import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from moic.utils import load_json
from PIL import Image

class EuclidData(Dataset):
    def __init__(self,split = "train"):
        super().__init__()
        self.root = ""
    
    def __len__(self):return 0

    def __getitem__(self,index):return index

class BattlecodeData(Dataset):
    def __init__(self,split = "train",data_path = None):
        super().__init__()
        
        assert split in ["train","val","test"]
        self.split = split
        self.root_dir = "/Users/melkor/Desktop/datasets/battlecode2"
        self.files = os.listdir(os.path.join(self.root_dir,split))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.question_file = load_json(os.path.join(self.root_dir,"{}_bc_qa.json".format(split)))

    def __len__(self): return 100
        #return len(self.question_file)

    def __getitem__(self,index):
        bind = self.question_file[index]
        image_id = bind["image"]
        image = Image.open(os.path.join(self.root_dir,self.split,"{}.jpg".format(image_id)))
        image = image.convert("RGB").resize([128,128])
        image = self.img_transform(image)
        return {"image":image,"program":bind["program"],"question":bind["question"],"answer":bind["answer"]}
