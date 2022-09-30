from euclidean_learner import *

mu = torch.tensor([100,56]).float()
sigma = 2 * torch.tensor([1,1])

normal = dists.Normal(mu,sigma)

x = torch.tensor([
    [1,2],[0,1]
])
normal_sample = normal.log_prob(x)
print(normal_sample)

print(normal.sample())

grid = make_grid((128,128)).permute([1,2,0])

pdf = normal.log_prob(grid)
pdf = torch.sum(pdf,-1).exp()

p1 = EuclidPointModel(mu)
pdf = p1.pdf(True).detach()
pdf = p1.pdf(False).detach()

plt.imshow(pdf,cmap="bone")
plt.show()
