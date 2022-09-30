from euclidean_learner import *

mu = torch.tensor([100,56]).float()

p1 = EuclidPointModel(mu)
pdf = p1.pdf(True).detach()
pdf = p1.pdf(False).detach()

plt.imshow(pdf,cmap="bone")
plt.show()
