import torch

# from nets import EncoderS

# class CNet(torch.nn.Module):

#     def __init__(self,
#                  nc : int,
#                  nz : int):

#         super(CNet, self).__init__()

#         self.net = EncoderS(1, nz, nc)

#     def forward(self,
#                 z : torch.Tensor,
#                 x : torch.Tensor) -> torch.Tensor:

#          return self.net(x, z)

from nets import Encoder

class CNet(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nz : int):

        super(CNet, self).__init__()

        self.net = Encoder(1, nc)
        self.lin = torch.nn.Linear(nz, nc, bias=False)

    def forward(self,
                z : torch.Tensor,
                x : torch.Tensor) -> torch.Tensor:

         return self.net(x) + self.lin(z)

class CModel(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nz : int,
                 step : float):

        super(CModel, self).__init__()

        self.nc = nc
        self.net = CNet(nc, nz)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=step)
        self.sheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda it: 100000 / (100000 + it))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.dummy_param = torch.nn.Parameter(torch.zeros([]), requires_grad=False)

    def optimize(self,
                 c : torch.Tensor,
                 z : torch.Tensor,
                 x : torch.Tensor,
                 l : torch.Tensor) -> tuple:
        
        c = c[l!=0]
        z = z[l!=0]
        x = x[l!=0]

        self.optimizer.zero_grad()
        scores = self.net(z, x)
        loss = self.criterion(scores, c).mean()
        loss.backward()
        self.optimizer.step()
        self.sheduler.step()
        
        return loss.detach()

    def sample(self,
               z : torch.Tensor,
               x : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            scores = self.net(z, x)
            distr = torch.distributions.one_hot_categorical.OneHotCategorical(logits=scores)
            return distr.sample().detach()

    def sample_random(self,
                      bs : int) -> torch.Tensor:
        
        ind = torch.randint(self.nc, [bs], device=self.dummy_param.device)
        return torch.nn.functional.one_hot(ind, num_classes=self.nc).float().detach()
    
    def inconsistency(self,
                      c : torch.Tensor,
                      z : torch.Tensor,
                      x : torch.Tensor):

        with torch.no_grad():
            scores = self.net(z, x)
            sampled = torch.distributions.one_hot_categorical.OneHotCategorical(logits=scores).sample()
            return (scores * (sampled - c)).sum(1).mean().detach(), sampled.detach()
