from euclidean_learner import *

mu = torch.tensor([100,56]).float()

p1 = EuclidPointModel(mu)
#pdf = p1.pdf(True).detach()
pdf = p1.pdf(False).detach()

#plt.imshow(pdf,cmap="bone")
#plt.show()

start  = mu
end    = mu * 0.5
test_line = segment(start,end,100)

print(test_line.shape)

grid = make_grid().permute([1,2,0])

print(grid.shape)

grid_expand = grid.flatten(start_dim = 0, end_dim = 1).unsqueeze(1).repeat([1,100,1])

diff = grid_expand - test_line 
min_diff = torch.min(diff,1).values

leng_diff = torch.norm(min_diff,dim = 1)


line_norm = dists.Normal(0,1)

logpdf = line_norm.log_prob(leng_diff)

logpdf = logpdf.view([128,128,1])

plt.imshow(logpdf.exp())
plt.show()