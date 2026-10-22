import torch

from nets import EncoderS

class ZNet(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nz : int):

        super(ZNet, self).__init__()

        self.net = EncoderS(1, nc, nz)

    def forward(self,
                c : torch.Tensor,
                x : torch.Tensor) -> torch.Tensor:
         
         return self.net(x, c)

class ZModel(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nz : int,
                 step : float):

        super(ZModel, self).__init__()

        self.nz = nz

        self.net = ZNet(nc, nz)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=step)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.sheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda it: 100000 / (100000 + it))

        self.dummy = torch.nn.Parameter(torch.zeros([]))

    def optimize(self,
                 c : torch.Tensor,
                 z : torch.Tensor,
                 x : torch.Tensor,
                 l : torch.Tensor) -> tuple:

        c = c[l!=1]
        z = z[l!=1]
        x = x[l!=1]

        self.optimizer.zero_grad()
        scores = self.net(c, x)
        loss = self.criterion(scores, z).sum(1).mean()
        loss.backward()
        self.optimizer.step()
        self.sheduler.step()

        return loss.detach()/(self.nz)

    def sample(self,
               c : torch.Tensor,
               x : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            scores = self.net(c, x)
            return torch.sigmoid(scores).bernoulli().detach()

    def sample_random(self,
                      bs : int) -> torch.Tensor:
        
        with torch.no_grad():
            return (torch.rand([bs, self.nz], device=self.dummy.device)<0.5).float().detach()

    def inconsistency(self,
                      c : torch.Tensor,
                      z : torch.Tensor,
                      x : torch.Tensor):

        with torch.no_grad():
            scores = self.net(c, x)
            sampled = torch.sigmoid(scores).bernoulli()
            return ((sampled - z) * scores).sum(1).mean().detach(), sampled.detach()
