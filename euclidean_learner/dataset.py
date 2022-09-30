import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from moic.utils import load_json
from PIL import Image

class EuclidData(Dataset):
    def __init__(self,split = "train",name = "ccc",resolution = (128,128)):
        super().__init__()
        assert split in ["train","test"],print("split {} not recognized.".format(split))
        self.root_dir = "geoclidean"
        self.concept_name = name
        self.split = split
        self.files = os.listdir(os.path.join(
            self.root_dir,"constraints","concept_{}".format(self.concept_name),
            self.split
        ))
        self.concept_path = os.path.join(
            self.root_dir,"constraints","concept_{}".format(self.concept_name),
            self.split
        )
        self.img_transform = transforms.Compose(
            [   
                transforms.ToTensor()]
        )
        self.question_file = None
        self.resolution = resolution

    def __len__(self):return len(self.files)

    def __getitem__(self,index):
        index = index + 1
        image = Image.open(os.path.join(self.concept_path,"{}_fin.png").format(index))
        image = self.img_transform(image.resize(self.resolution))
        return {"image":image}

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
