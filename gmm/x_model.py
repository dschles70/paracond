import torch

# "common_sigma" controls, whether the standard deviations 
# should be equal (when True) or free (when False)
class XNet(torch.nn.Module):

    def __init__(self,
                 common_sigma: bool):

        super(XNet, self).__init__()

        self.mu = torch.nn.Linear(1, 1)
        self.common_sigma = common_sigma
        if common_sigma:
            self.lsigma = torch.nn.Parameter(torch.zeros([]))
        else:
            self.lsigma = torch.nn.Linear(1, 1)

    def forward(self,
                c : torch.Tensor):

        if self.common_sigma:
            return self.mu(c), self.lsigma
        else:
            return self.mu(c), self.lsigma(c)

class XModel(torch.nn.Module):

    def __init__(self,
                 step : float,
                 common_sigma: bool):

        super(XModel, self).__init__()

        self.net = XNet(common_sigma)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=step)

    def optimize(self,
                 c : torch.Tensor,
                 x : torch.Tensor,
                 l : torch.Tensor):

        c = c[l!=1]
        x = x[l!=1]

        self.optimizer.zero_grad()
        mu, lsigma = self.net(c)
        sigma = torch.exp(lsigma)
        loss = 0.91894 + lsigma + ((x - mu)**2) / (2*(sigma**2))
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def sample(self,
               c : torch.Tensor):

        with torch.no_grad():
            mu, lsigma = self.net(c)
            sigma = torch.exp(lsigma)
            return (torch.randn_like(mu) * sigma + mu).detach()

    def get_ms(self,
               c : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            mu, lsigma = self.net(c)
            sigma = torch.exp(lsigma)
            return mu.detach(), sigma.detach()

    # provides "log p(x'|c) - log p(x|c)", where x' is sampled,
    # used in the DBC-measure, see inconsistency.py for details
    def inconsistency(self,
                      c : torch.Tensor,
                      x : torch.Tensor):

        with torch.no_grad():
            mu, lsigma = self.net(c)
            sigma = torch.exp(lsigma)
            x_s = torch.randn_like(mu) * sigma + mu
            l = (x - mu) ** 2
            l_s = (x_s - mu) ** 2
            return ((l - l_s) / (2*(sigma**2))).mean()
