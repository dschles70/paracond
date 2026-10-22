# implements p(x|c,z)
# the structure is very similar to others
# see c_model.py for more detailed comments

import torch
from nets import Decoder

class XNet(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nz : int):

        super(XNet, self).__init__()

        self.net = Decoder(nc + nz, 1)

    def forward(self,
                c : torch.Tensor,
                z : torch.Tensor) -> torch.Tensor:
         
         cz = torch.cat((c, z), dim=1)
         return self.net(cz)

class XModel(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nz : int,
                 step : float):

        super(XModel, self).__init__()

        self.net = XNet(nc, nz)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=step)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
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
        scores = self.net(c, z)
        loss = self.criterion(scores, x).sum([1,2,3]).mean()
        loss.backward()
        self.optimizer.step()
        self.sheduler.step()

        return loss.detach()/(28*28)

    def sample(self,
               c : torch.Tensor,
               z : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            scores = self.net(c, z)
            return torch.sigmoid(scores).bernoulli().detach()

    # used for visualization only
    def get_probs(self,
                  c : torch.Tensor,
                  z : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            scores = self.net(c, z)
            return torch.sigmoid(scores).detach()
