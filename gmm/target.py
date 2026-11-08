import torch

from c_model import CModel
from x_model import XModel
from data_generator import DataGenerator

class Target():
    def __init__(self,
                 c_model : CModel,
                 x_model : XModel,
                 bs_a: int,
                 bs_n: int):

        self.c_model = c_model
        self.x_model = x_model

        self.bs_a = bs_a
        self.bs_n = bs_n

        self.data_geneator = DataGenerator(bs_n)

        self.c, self.x, self.l = self.get_new_data()
        while True:
            c, x, l = self.get_new_data()
            self.c = torch.cat((self.c, c))
            self.x = torch.cat((self.x, x))
            self.l = torch.cat((self.l, l))
            if self.c.shape[0] >= self.bs_a:
                break
        
        self.c = self.c[:self.bs_a]
        self.x = self.x[:self.bs_a]
        self.l = self.l[:self.bs_a]

        for _ in range(100):
            self.get_batch()

    def get_new_data(self):

        x = self.data_geneator.get_batch()

        c = self.c_model.sample(x)

        l = torch.zeros([x.shape[0]]).long()

        return c, x, l

    def get_batch(self):

        n_to_keep = self.bs_a - self.bs_n

        # permute, keep a part
        perm = torch.randperm(self.bs_a)
        c_ch = torch.split(self.c[perm], [n_to_keep, self.bs_n])[0]
        x_ch = torch.split(self.x[perm], [n_to_keep, self.bs_n])[0]
        l_ch = torch.split(self.l[perm], [n_to_keep, self.bs_n])[0]

        # split
        c_ch = list(torch.chunk(c_ch, 2))
        x_ch = list(torch.chunk(x_ch, 2))
        l_ch = list(torch.chunk(l_ch, 2))

        # resample
        c_ch[0] = self.c_model.sample(x_ch[0])
        l_ch[0] = torch.zeros([x_ch[0].shape[0]]).long()
        x_ch[1] = self.x_model.sample(c_ch[1])
        l_ch[1] = torch.ones([x_ch[1].shape[0]]).long()

        # get new data from the training set
        c, x, l = self.get_new_data()

        # put together
        self.c = torch.cat([c] + c_ch)
        self.x = torch.cat([x] + x_ch)
        self.l = torch.cat([l] + l_ch)

        return self.c, self.x, self.l
