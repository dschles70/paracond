import torch
from nets import Decoder

class XNet(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nz : int):

        super(XNet, self).__init__()

        self.net = Decoder(nc + nz, 1)
        self.lsigma = torch.nn.Parameter(torch.zeros([]))

    def forward(self,
                c : torch.Tensor,
                z : torch.Tensor) -> torch.Tensor:
         
         cz = torch.cat((c, z), dim=1)
         return self.net(cz), self.lsigma

class XModel(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nz : int,
                 step : float):

        super(XModel, self).__init__()

        self.net = XNet(nc, nz)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=step)
        self.sheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda it: 100000 / (100000 + it))

    def optimize(self,
                 c : torch.Tensor,
                 z : torch.Tensor,
                 x : torch.Tensor,
                 l : torch.Tensor) -> tuple:

        c = c[l!=2]
        z = z[l!=2]
        x = x[l!=2]

        self.optimizer.zero_grad()
        mu, lsigma = self.net(c, z)
        sigma = torch.exp(lsigma)
        loss = 0.91894 + lsigma + ((x - mu)**2) / (2*(sigma**2))
        loss = loss.sum([1,2,3]).mean()
        loss.backward()
        self.optimizer.step()
        self.sheduler.step()

        return loss.detach()/(28*28)

    def sample(self,
               c : torch.Tensor,
               z : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            mu, lsigma = self.net(c, z)
            sigma = torch.exp(lsigma)
            return (torch.randn_like(mu) * sigma + mu).detach()

    def get_ms(self,
               c : torch.Tensor,
               z : torch.Tensor):

        with torch.no_grad():
            mu, lsigma = self.net(c, z)
            sigma = torch.exp(lsigma)
            return mu.detach(), sigma.detach()
