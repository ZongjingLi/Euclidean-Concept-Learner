from euclidean_learner import *

import matplotlib.pyplot as plt

if __name__ == "__main__":
    concept_dataset = EuclidData("train",name = "ccc")
    print(len(concept_dataset))
    
    dataloader = DataLoader(concept_dataset,batch_size = 1)
    
    plt.ion()
    for sample in dataloader:
        image = sample["image"][0].permute([1,2,0])
        plt.cla()
        plt.imshow(image)
        plt.pause(1)
        
        mu = torch.tensor([100,56]).float()
        p1 = EuclidPointModel(mu)
        pdf1 = p1.pdf(False).detach()

        plt.imshow(pdf1,cmap = "bone")
        plt.pause(.1)
        
        # evaluate the second point
        p2 = EuclidPointModel(mu * 0.3)
        pdf2 = p2.pdf(False).detach()

        plt.imshow(pdf2,cmap = "bone")
        plt.pause(.1)

        line1 = EuclidLineModel(p1,p2)
        pdf3 = line1.pdf(False).detach()

        plt.imshow(pdf3*0.1 + pdf1 + pdf2,cmap = "bone")

        plt.pause(1)

        adjust_model_to_observation(line1,image,50,True)
    
    plt.ioff()
    plt.show()

    """
            # p1 existence map
        evl = p1.exist(image,True).detach()

        plt.imshow(evl,cmap = "bone")
        plt.pause(1)
    """
    