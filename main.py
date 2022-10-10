
from geometric_learner.model import *
from geometric_learner.primitive import *

import argparse

train_parser = argparse.ArgumentParser()
train_parser.add_argument("--lr",default = 1e-2, type = float, help = "learning rate of")
train_opt = train_parser.parse_args(args = [])

if __name__ == "__main__":

    # prepare the dataset    
    dataset = EuclidConceptData("train","angle")
    loader  = DataLoader(dataset,batch_size = 1,shuffle = True)

    # prepare the geometric autoencoder model
    model  = GeometricAutoEncoder(model_opt)

    print(model);plt.ion()
    optim = torch.optim.Adam(model.parameters(), lr = 2e-4)
    # draw samples from the dataloader
    for epoch in range(1000):
        loss = 0
        optim.zero_grad()
        for sample in loader:
            # execute the GeoAutoencoder
            programs = [term[0] for term in sample["programs"]]
            outputs = model(sample["image"],programs)
            loss = loss + torch.sum(torch.binary_cross_entropy_with_logits(outputs,sample["image"]))
        
        loss.backward()
        optim.step()
        print(loss)
        # plot the concept structure found
        g = model.decoder.struct
        
        plt.figure("concept dag");
        plt.cla();nx.draw_networkx(g)
        plt.figure("inputs vs recons")
        plt.subplot(1,2,1);plt.cla();plt.imshow(sample["image"][0].permute([1,2,0]))
        plt.subplot(1,2,2);plt.cla();plt.imshow(outputs.permute([1,2,0]).exp().detach())
        #nx.draw_shell(g,with_labels = True)
        plt.pause(1)
