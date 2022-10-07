
from geometric_learner.model import *

dataset = EuclidConceptData("angle")
for sample in dataset:
    print(sample["programs"])