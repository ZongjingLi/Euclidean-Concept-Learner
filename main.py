
from geometric_learner.model import *
from geometric_learner.primitive import *

dataset = EuclidConceptData("train","diameter")
loader = DataLoader(dataset,batch_size = 1)
struct = GeometricStructure()

for sample in loader:

    programs = [term[0] for term in sample["programs"]]
    g = struct.make_dag(programs)
    nx.draw_networkx(g)
    #nx.draw_shell(g,with_labels = True)
    plt.show()
