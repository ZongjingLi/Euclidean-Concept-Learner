from euclidean_learner import *

import matplotlib.pyplot as plt

if __name__ == "__main__":
    concept_dataset = EuclidData("train",name = "ccc")
    print(len(concept_dataset))
    
    dataloader = DataLoader(concept_dataset,batch_size = 1)
    
    plt.ion()
    for sample in dataloader:
        image = sample["image"]
        plt.imshow(image[0].permute([1,2,0]))
        plt.pause(1)
    plt.ioff()
    plt.show()