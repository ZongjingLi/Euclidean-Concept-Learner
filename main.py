
from geometric_learner.model import *

dataset = EuclidConceptData("train","angle")
loader = DataLoader(dataset,batch_size = 1)

for sample in loader:
    print(sample["programs"])