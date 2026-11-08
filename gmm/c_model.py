import torch

class CNet(torch.nn.Module):

    def __init__(self):

        super(CNet, self).__init__()

        self.lin = torch.nn.Linear(1, 1)

    def forward(self,
                x : torch.Tensor):
         
         return self.lin(x)

class CModel(torch.nn.Module):

    def __init__(self,
                 step : float):

        super(CModel, self).__init__()

        self.net = CNet()

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=step)

    def optimize(self,
                 c : torch.Tensor,
                 x : torch.Tensor,
                 l : torch.Tensor):

        c = c[l!=0]
        x = x[l!=0]

        self.optimizer.zero_grad()
        scores = self.net(x)
        loss = self.criterion(scores, c).mean()
        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def sample(self,
               x : torch.Tensor):

        with torch.no_grad():
            scores = self.net(x)
            return torch.sigmoid(scores).bernoulli().detach()

    def sample_random(self,
                      bs : int):

            return (torch.rand([bs, 1]) < 0.5).float()

    def inconsistency(self,
                      c : torch.Tensor,
                      x : torch.Tensor):

        with torch.no_grad():
            scores = self.net(x)
            c_s = torch.sigmoid(scores).bernoulli()
            return ((c_s - c) * scores).mean()
    