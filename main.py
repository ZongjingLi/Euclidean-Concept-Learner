
from geometric_learner.model import *

dataset = EuclidConceptData("train","angle")
for sample in dataset:
    print(sample["programs"])