from euclidean_learner import *

import matplotlib.pyplot as plt

if __name__ == "__main__":
    concept_dataset = EuclidData("train",name = "ccc")
    print(len(concept_dataset))
    
    dataloader = DataLoader(concept_dataset,batch_size = 1)
    
    plt.ion()
    for sample in dataloader:
        image = sample["image"][0].permute([1,2,0])
        print(image.shape)
        plt.cla()
        plt.imshow(image)
        plt.pause(1)
        
        mu = torch.tensor([100,56]).float()
        p1 = EuclidPointModel(mu)
        pdf = p1.pdf(False).detach()

        plt.imshow(pdf,cmap="bone")
        plt.pause(1)

        evl = p1.exist(image).detach()
        print(evl.shape)
        plt.imshow(evl)
        plt.pause(1)

    plt.ioff()
    plt.show()