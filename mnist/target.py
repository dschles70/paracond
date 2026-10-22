import torch

from c_model import CModel
from z_model import ZModel
from x_model import XModel

from mnist import MNIST

class Target():
    def __init__(self,
                 c_model : CModel,
                 z_model : ZModel,
                 x_model : XModel,
                 bs_a: int, # all
                 bs_n: int, # new
                 device: int):

        self.c_model = c_model
        self.z_model = z_model
        self.x_model = x_model

        self.bs_a = bs_a
        self.bs_n = bs_n
        self.device = device

        self.mnist = MNIST(bs_n)

        self.c, self.z, self.x, self.l = self.get_new_data()
        while True:
            c, z, x, l = self.get_new_data()
            self.c = torch.cat((self.c, c))
            self.z = torch.cat((self.z, z))
            self.x = torch.cat((self.x, x))
            self.l = torch.cat((self.l, l))
            if self.c.shape[0] >= self.bs_a:
                break
        
        self.c = self.c[:self.bs_a]
        self.z = self.z[:self.bs_a]
        self.x = self.x[:self.bs_a]
        self.l = self.l[:self.bs_a]

        for _ in range(100):
            self.get_batch()

    def get_new_data(self):

        c, x = self.mnist.get_train(self.device)
        # add noise to MNIST
        mask = (torch.rand_like(x) < 0.01).float()
        x = x * (1 - mask) + (1 - x) * mask

        z = self.z_model.sample(c, x)

        l = torch.ones([x.shape[0]], device=self.device).long()

        return c, z, x, l

    def get_batch(self):

        n_to_keep = self.bs_a - self.bs_n

        # permute, keep a part
        perm = torch.randperm(self.bs_a)
        c_ch = torch.split(self.c[perm], [n_to_keep, self.bs_n])[0]
        z_ch = torch.split(self.z[perm], [n_to_keep, self.bs_n])[0]
        x_ch = torch.split(self.x[perm], [n_to_keep, self.bs_n])[0]
        l_ch = torch.split(self.l[perm], [n_to_keep, self.bs_n])[0]

        # split
        c_ch = list(torch.chunk(c_ch, 3))
        z_ch = list(torch.chunk(z_ch, 3))
        x_ch = list(torch.chunk(x_ch, 3))
        l_ch = list(torch.chunk(l_ch, 3))

        # resample
        c_ch[0] = self.c_model.sample(z_ch[0], x_ch[0])
        l_ch[0] = torch.zeros([c_ch[0].shape[0]], device=self.device).long()
        z_ch[1] = self.z_model.sample(c_ch[1], x_ch[1])
        l_ch[1] = torch.ones([z_ch[1].shape[0]], device=self.device).long()
        x_ch[2] = self.x_model.sample(c_ch[2], z_ch[2])
        l_ch[2] = (torch.ones([x_ch[2].shape[0]], device=self.device) * 2).long()

        # get new data from the training set
        c, z, x, l = self.get_new_data()

        # put together
        self.c = torch.cat([c] + c_ch)
        self.z = torch.cat([z] + z_ch)
        self.x = torch.cat([x] + x_ch)
        self.l = torch.cat([l] + l_ch)

        return self.c, self.z, self.x, self.l
