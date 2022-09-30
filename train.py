from euclidean_learner import *

if __name__ == "__main__":
    concept_dataset = EuclidData("train",name = "ccc")
    print(len(concept_dataset))
    
    dataloader = DataLoader(concept_dataset,batch_size = 1)
    
    for sample in dataloader:
        print(sample)