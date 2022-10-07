
from geometric_learner.model import *
from geometric_learner.primitive import *

dataset = EuclidConceptData("train","angle")
loader = DataLoader(dataset,batch_size = 1)
struct = GeometricStructure()

for sample in loader:
    for t in sample["programs"]:print(t)
    programs = [term[0] for term in sample["programs"]]
    struct.make_dag(programs)
    print("")