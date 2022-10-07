
from geometric_learner.model import *
from geometric_learner.primitive import *

import argparse

train_parser = argparse.ArgumentParser()
train_parser.add_argument("--lr",default = 1e-2, type = float, help = "learning rate of")
train_opt = train_parser.parse_args(args = [])

if __name__ == "__main__":
    
    dataset = EuclidConceptData("train","diameter")
    loader = DataLoader(dataset,batch_size = 1)

    struct = GeometricStructure()

    for sample in loader:

        programs = [term[0] for term in sample["programs"]]
        g = struct.make_dag(programs)
        nx.draw_networkx(g)
        #nx.draw_shell(g,with_labels = True)
        plt.show()
