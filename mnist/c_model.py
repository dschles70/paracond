import torch
from nets import Encoder

# a wrapper for the network implementing p(c|z,x)
class CNet(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nz : int):

        super(CNet, self).__init__()

        self.net = Encoder(1, nc)
        self.lin = torch.nn.Linear(nz, nc, bias=False)

    # returns logits (remember, c is a categorical variable)
    def forward(self,
                z : torch.Tensor,
                x : torch.Tensor) -> torch.Tensor:

         return self.net(x) + self.lin(z)

class CModel(torch.nn.Module):

    def __init__(self,
                 nc : int, # the number of classes (10 for MNIST)
                 nz : int, # the number of the latent dimensions
                 step : float):

        super(CModel, self).__init__()

        self.nc = nc
        self.net = CNet(nc, nz)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=step)
        # we use harmonic scheduler (just for the stability), 
        # the decay of which is chosen so that after 1M iterations
        # the gradient step size is decreased by the factor 11
        self.sheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda it: 100000 / (100000 + it))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        # necessary only to store "device"
        self.dummy_param = torch.nn.Parameter(torch.zeros([]), requires_grad=False)

    # gradient update
    def optimize(self,
                 c : torch.Tensor, # classes (target)
                 z : torch.Tensor, # latents
                 x : torch.Tensor, # image
                 l : torch.Tensor):

        # filter out samples, where c was just sampled (see "target.py" for details)
        c = c[l!=0]
        z = z[l!=0]
        x = x[l!=0]

        # optimize (standard)
        self.optimizer.zero_grad()
        scores = self.net(z, x)
        loss = self.criterion(scores, c).mean()
        loss.backward()
        self.optimizer.step()
        self.sheduler.step()
        
        return loss.detach()

    # sampling from p(c|z,x)
    def sample(self,
               z : torch.Tensor,
               x : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            scores = self.net(z, x)
            distr = torch.distributions.one_hot_categorical.OneHotCategorical(logits=scores)
            return distr.sample().detach()

    # sample uniformly (can be used e.g. for initialization, i.e. starting MCMC)
    def sample_random(self,
                      bs : int) -> torch.Tensor:
        
        ind = torch.randint(self.nc, [bs], device=self.dummy_param.device)
        return torch.nn.functional.one_hot(ind, num_classes=self.nc).float().detach()
