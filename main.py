from euclidean_learner import *

mu = torch.tensor([1,2]).float()
sigma = 0.1 * torch.tensor([1,1])

normal = dists.Normal(mu,sigma)

x = torch.tensor([
    [1,2],[0,1]
])
normal_sample = normal.log_prob(x)
print(normal_sample)

print(normal.sample())


grid = make_grid((3,3))

print(grid.shape)