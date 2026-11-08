import torch

class DataGenerator():

    def __init__(self,
                 nsamples : int):
        
        self.nsamples = nsamples

        self.w = torch.as_tensor([[0.8, 0.2]])
        self.mu = torch.as_tensor([[-0.5, 0.5]])
        self.sigma = torch.as_tensor([[0.5, 0.1]])

        # tabulate the probabilty density
        x = torch.linspace(-2, 2, 1000)
        p0 = torch.log(self.sigma[0,0]) + ((x - self.mu[0,0])**2) / (2*(self.sigma[0,0]**2))
        p0 = self.w[0,0] * torch.exp(-p0)
        p1 = torch.log(self.sigma[0,1]) + ((x - self.mu[0,1])**2) / (2*(self.sigma[0,1]**2))
        p1 = self.w[0,1] * torch.exp(-p1)
        p = p0 + p1
        self.p = p / p.sum()
        self.lp = torch.log(self.p)

    def get_batch(self):
        probs = self.w.expand([self.nsamples,2])
        c = torch.distributions.one_hot_categorical.OneHotCategorical(probs = probs).sample()
        mu = (self.mu.expand_as(c) * c).sum(1, keepdim=True)
        sigma = (self.sigma.expand_as(c) * c).sum(1, keepdim=True)

        return torch.randn([self.nsamples,1]) * sigma + mu

    def kl(self,
           q : torch.Tensor):

        return (self.p * (self.lp - torch.log(q))).sum()
