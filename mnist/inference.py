import os
import time
import argparse

import torch
from c_model import CModel
from z_model import ZModel
from mnist import MNIST

from helpers import load_model

def inference(x : torch.Tensor,
              c_model : CModel,
              z_model : ZModel):

    bs = x.shape[0]
    z = z_model.sample_random(bs)
    cmarg = torch.zeros([bs, c_model.nc], device=x.device)

    for _ in range(100):
        with torch.no_grad():
            scores = c_model.net(z, x)
            probs = torch.softmax(scores, dim=1)
            cmarg = cmarg + probs
            c = torch.distributions.one_hot_categorical.OneHotCategorical(probs=probs).sample().detach()
        z = z_model.sample(c, x)
    cdec = torch.argmax(cmarg, dim=1)
    return torch.nn.functional.one_hot(cdec, num_classes=c.shape[1]).float()

if __name__ == '__main__':
    print(os.uname(), flush=True)
    time0 = time.time()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_prefix', required=True)
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

    loadprot = load_model(c_model, './models/c_' + args.model_prefix + '.pt', strict=True, load_optimizer=False)
    print('c: ', loadprot, flush=True)
    loadprot = load_model(z_model, './models/z_' + args.model_prefix + '.pt', strict=True, load_optimizer=False)
    print('z: ', loadprot, flush=True)

    mnist = MNIST(bs)

    x_all = torch.split(mnist.x_train.to(device), bs)
    c_all = torch.split(mnist.c_train.to(device), bs)
    n_batches = len(x_all)

    total_acc = 0
    print(n_batches, 'batches in the training set', flush=True)
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
    print('Training accuracy: ', (total_acc/mnist.x_train.shape[0]).cpu().numpy(), flush=True)

    x_all = torch.split(mnist.x_val.to(device), bs)
    c_all = torch.split(mnist.c_val.to(device), bs)
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
    print('Validation accuracy: ', (total_acc/mnist.x_val.shape[0]).cpu().numpy(), flush=True)
