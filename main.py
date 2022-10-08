
from geometric_learner.model import *
from geometric_learner.primitive import *

import argparse

train_parser = argparse.ArgumentParser()
train_parser.add_argument("--lr",default = 1e-2, type = float, help = "learning rate of")
train_opt = train_parser.parse_args(args = [])

if __name__ == "__main__":

    # prepare the dataset    
    dataset = EuclidConceptData("train","radii")
    loader  = DataLoader(dataset,batch_size = 1)

    # prepare the geometric autoencoder model
    model  = GeometricAutoEncoder(model_opt)

    print(model)
    
    # draw samples from the dataloader
    for sample in loader:
        # execute the GeoAutoencoder
        programs = [term[0] for term in sample["programs"]]
        outputs = model(sample["image"],programs)

        # plot the concept structure found
        g = model.decoder.struct
        nx.draw_networkx(g)
        #nx.draw_shell(g,with_labels = True)
        plt.show()
