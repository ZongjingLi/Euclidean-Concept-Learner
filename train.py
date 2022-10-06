from euclidean_learner import *

import matplotlib.pyplot as plt

if __name__ == "__main__":
    concept_dataset = EuclidData("train",name = "ccc")
    concept_dataset = EuclidConceptData("train",name = "angle")
    print(len(concept_dataset))
    
    dataloader = DataLoader(concept_dataset,batch_size = 1)
    
    plt.ion()
    for sample in dataloader:
        image = sample["image"][0].permute([1,2,0])
        plt.cla()
        plt.imshow(image)
        plt.pause(1)
        
        mu = 64 + torch.randn([2]).float()
        p1 = EuclidPointModel(10* torch.randn([2]) + torch.tensor([32,32]))
        p2 = EuclidPointModel(10* torch.randn([2]) + torch.tensor([32,32]))
        p3 = EuclidPointModel(10* torch.randn([2]) + torch.tensor([32,32]))
        line1 = EuclidLineModel(p1,p2)
        angle = EuclidAngleModel(p1,p2,p3)
    
        adjust_model_to_observation(angle,image,300,True)
    
    plt.ioff()
    plt.show()
    print("done")
    """
            # p1 existence map
        evl = p1.exist(image,True).detach()

        plt.imshow(evl,cmap = "bone")
        plt.pause(1)
    """
    