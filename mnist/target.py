import torch

from c_model import CModel
from z_model import ZModel
from x_model import XModel

from my_mnist import MY_MNIST

# Implements "blending" learning as described in the paper at the end of section 4.
# There is a batch containing samples which are distributed along the generated chain.
# At each iteration (see get_batch() below), the batch is updated.
# The learning procedure (in the main.py) optimizes all conditionals on the whole batch.

class Target():
    def __init__(self,
                 c_model : CModel, # p(c|z,x)
                 z_model : ZModel, # p(z|c,x)
                 x_model : XModel, # p(x|c,z)
                 bs_a: int, # the total number of samples in the batch
                 bs_n: int, # the number of the newly injected samples
                 device: int # the device
                 ):

        self.c_model = c_model
        self.z_model = z_model
        self.x_model = x_model

        self.bs_a = bs_a
        self.bs_n = bs_n
        self.device = device

        # the dataset
        self.my_mnist = MY_MNIST(bs_n)

        # initialize the batch
        # c -- classes
        # z -- latent variables
        # x -- images
        # l -- for each sample the indicator (0/1/2), showing what component was sampled last

        # first, get everything from the training data (completed)
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

        # second, perform updates several times in order to have the samples
        # properly distributed along the chain
        for _ in range(100):
            self.get_batch()

    # newly injected data
    def get_new_data(self):

        # get the new data from the dateset
        c, x = self.my_mnist.get_train(self.device)

        # we add a very small portion of noise here (remember, MNIST is binarized)
        # otherwise we have a couple of pixel positions, where the pixels are just always black,
        # which leads to numerical instabilities
        mask = (torch.rand_like(x) < 0.01).float()
        x = x * (1 - mask) + (1 - x) * mask

        # completion consists of just sampling once from p(z|c,x)
        z = self.z_model.sample(c, x)

        # the last sampled is z, corresponds to "1"
        l = torch.ones([x.shape[0]], device=self.device).long()

        return c, z, x, l

    def get_batch(self):

        n_to_keep = self.bs_a - self.bs_n

        # shuffle, keep a part
        perm = torch.randperm(self.bs_a)
        c_ch = torch.split(self.c[perm], [n_to_keep, self.bs_n])[0]
        z_ch = torch.split(self.z[perm], [n_to_keep, self.bs_n])[0]
        x_ch = torch.split(self.x[perm], [n_to_keep, self.bs_n])[0]
        l_ch = torch.split(self.l[perm], [n_to_keep, self.bs_n])[0]

        # hence, at each iteration, each example in the batch disappears with probability "bs_n / bs_a"
        # consequently, the average chain length 
        # (i.e. the average number of iteration a sample stays in the batch and is resampled)
        # is "bs_a / bs_n"

        # split into 3 chunks for resampling c, z or x
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
        # the indicators l show, which component was re-samples last
        # 0 -- c, 1 -- z, and 2 -- x

        # get new data from the training set (completed)
        c, z, x, l = self.get_new_data()

        # put together
        self.c = torch.cat([c] + c_ch)
        self.z = torch.cat([z] + z_ch)
        self.x = torch.cat([x] + x_ch)
        self.l = torch.cat([l] + l_ch)

        return self.c, self.z, self.x, self.l
