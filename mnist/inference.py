import os
import time
import argparse

import torch
from c_model import CModel
from z_model import ZModel
from my_mnist import MY_MNIST

from helpers import load_model

# this function is imported from the "main.py" to watch the accuracies "on the fly"
def inference(x : torch.Tensor, # images
              c_model : CModel, # p(c|z,x)
              z_model : ZModel  # p(z|c,x)
              ):

    bs = x.shape[0]

    # init z and the marginals for classes
    z = z_model.sample_random(bs)
    cmarg = torch.zeros([bs, c_model.nc], device=x.device)

    for _ in range(100):
        # we accumulate probabilities instead of the sampled values
        # should be just more accurate
        with torch.no_grad():
            scores = c_model.net(z, x)
            probs = torch.softmax(scores, dim=1) # conditional probabilities
            cmarg = cmarg + probs # accumulation

            # sampling from p(c|z,x)
            c = torch.distributions.one_hot_categorical.OneHotCategorical(probs=probs).sample().detach()

        # sampling from p(z|c,x)
        z = z_model.sample(c, x)

    # maximum marginal decision
    cdec = torch.argmax(cmarg, dim=1)
    return torch.nn.functional.one_hot(cdec, num_classes=c.shape[1]).float()

# perform inference for the whole dataset
if __name__ == '__main__':
    print(os.uname(), flush=True)
    time0 = time.time()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--nz', type=int, required=True)

    args = parser.parse_args()

    bs = 1000

    # printout
    print('# Starting at ' + time.strftime('%c'), flush=True)

    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    # numbers of hidden variables
    nz = args.nz
    nc = 10

    c_model = CModel(nc, nz, 0).to(device)
    z_model = ZModel(nc, nz, 0).to(device)

    loadprot = load_model(c_model, './models/c_' + args.id + '.pt', strict=True, load_optimizer=False)
    print('c: ', loadprot, flush=True)
    loadprot = load_model(z_model, './models/z_' + args.id + '.pt', strict=True, load_optimizer=False)
    print('z: ', loadprot, flush=True)

    my_mnist = MY_MNIST(bs)

    # training set
    x_all = torch.split(my_mnist.x_train.to(device), bs)
    c_all = torch.split(my_mnist.c_train.to(device), bs)
    n_batches = len(x_all)

    total_acc = 0
    print(n_batches, 'batches in the training set', flush=True)
    for b in range(n_batches):
        c = c_all[b].to(device)
        x = x_all[b].to(device)
        dec = inference(x, c_model, z_model)
        c = torch.argmax(c, dim=1) # to make it class number from one-hot encoding
        dec = torch.argmax(dec, dim=1)
        acc = (c == dec).float().sum()
        total_acc += acc

        print('.', end='', flush=True)

    print()
    print('Training accuracy: ', (total_acc/my_mnist.x_train.shape[0]).cpu().numpy(), flush=True)

    # validation set
    x_all = torch.split(my_mnist.x_val.to(device), bs)
    c_all = torch.split(my_mnist.c_val.to(device), bs)
    n_batches = len(x_all)

    total_acc = 0
    print(n_batches, 'batches in the validation', flush=True)
    for b in range(n_batches):
        c = c_all[b].to(device)
        x = x_all[b].to(device)
        dec = inference(x, c_model, z_model)
        c = torch.argmax(c, dim=1)
        dec = torch.argmax(dec, dim=1)
        acc = (c == dec).float().sum()
        total_acc += acc

        print('.', end='', flush=True)

    print()
    print('Validation accuracy: ', (total_acc/my_mnist.x_val.shape[0]).cpu().numpy(), flush=True)
