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

grid = make_grid().permute([1,2,0])

grid_expand = grid.flatten(start_dim = 0, end_dim = 1).unsqueeze(0).repeat([100,1,1])

"""
def harmard_min(source,set):
    outputs = []
    for i in range(source.size(0)):
        outputs.append(torch.min(source - set,-1).values())
    return torch.stack(outputs)
"""

diff = grid_expand - test_line.unsqueeze(1).repeat([1,16384,1])
print(diff.shape)
leng_diff = torch.norm(diff,dim = -1)

print(leng_diff.shape)

min_diff = torch.min(leng_diff,0).values

print(min_diff.shape)

plt.plot(min_diff)
plt.show()

line_norm = dists.Normal(0,1)

logpdf = line_norm.log_prob(min_diff)


logpdf = logpdf.view([128,128,1])

plt.imshow(logpdf.exp())
plt.show()