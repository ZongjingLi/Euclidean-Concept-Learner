# Euclid Concept Learner
 This repo is mainly used for solving the problem proposed by the paper “Geoclidean”. We propose the neural-heusitic search model for geometric concepts.

# Background 
The background of the repo is about a few-shot learning problem: given few pictures of a geometric concept,
how can the model learn it and generalize it to other tasks (discriminative). In this repo, a model is created
to learn the concept model from observation.

# Model
The euclid concept learner model consists of three components, 1) an image encoder that characterize information

# Experiments

## failed experiments
The first thought to write such a model will be a neual-heusitc search based on the possiblity of an input image is a realization of concept model $c$: $Pr[c|x]$. One may want to use the method of soft line and points to evaluate the probability or cross-entropy of the input image is explained by concept model. However, there are several problems during the test phase.  