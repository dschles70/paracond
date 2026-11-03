import os
import time
import argparse
import torchvision.utils as vutils

import torch
from c_model import CModel
from z_model import ZModel
from x_model import XModel
from my_fmnist import MY_FMNIST

from helpers import load_model, generate_seq

if __name__ == '__main__':
    print(os.uname(), flush=True)
    time0 = time.time()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id',    type=str,   required=True)
    parser.add_argument('--nz',    type=int,   required=True)
    parser.add_argument('--noise', type=float, required=True)

    args = parser.parse_args()

    bs = 1000

    # printout
    print('# Starting at ' + time.strftime('%c'), flush=True)

    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    # numbers of hidden variables
    nz = args.nz
    nc = 10

    # create models
    c_model = CModel(nc, nz, 0).to(device)
    z_model = ZModel(nc, nz, 0).to(device)
    x_model = XModel(nc, nz, 0).to(device)

    # load models
    loadprot = load_model(c_model, './models/c_' + args.id + '.pt', strict=True, load_optimizer=False)
    print('c: ', loadprot, flush=True)
    loadprot = load_model(z_model, './models/z_' + args.id + '.pt', strict=True, load_optimizer=False)
    print('z: ', loadprot, flush=True)
    loadprot = load_model(x_model, './models/x_' + args.id + '.pt', strict=True, load_optimizer=False)
    print('x: ', loadprot, flush=True)

    # prepare the dataset
    my_fmnist = MY_FMNIST(6)

    # for visualization
    mean = my_fmnist.mean.to(device)
    std = my_fmnist.std.to(device)

    # split the validation set into batches
    x_all = torch.split(my_fmnist.x_val.to(device), bs)
    c_all = torch.split(my_fmnist.c_val.to(device), bs)
    n_batches = len(x_all)

    # go through the validation set, compute and accumulate the accuracy
    total_acc = 0
    print(n_batches, 'batches in the validation set', flush=True)
    for b in range(n_batches):
        c0 = c_all[b].to(device)
        x0 = x_all[b].to(device)
        mask = (torch.rand_like(x0) < args.noise).float()

        bs = c0.shape[0]
        c1 = c_model.sample_random(bs)
        cmarg = torch.zeros_like(c1)
        z1 = z_model.sample_random(bs)
        x1 = x_model.sample(c1, z1) * mask + x0 * (1 - mask)

        # this generates a sequence of operations without repetitions
        ops = generate_seq(100, 3)

        # generate, accumulate marginals (unnormalized)
        for op in ops:
            if op == 0:
                with torch.no_grad():
                    scores = c_model.net(z1, x1)
                    probs = torch.softmax(scores, dim=1)
                    cmarg = cmarg + probs
                    c1 = torch.distributions.one_hot_categorical.OneHotCategorical(probs=probs).sample().detach()
            elif op == 1:
                z1 = z_model.sample(c1, x1)
            else:
                x1 = x_model.sample(c1, z1) * mask + x0 * (1 - mask)

        # decisions and accuracies
        cdec = torch.argmax(cmarg, dim=1)
        c0 = torch.argmax(c0, dim=1)
        acc = (c0 == cdec).float().sum()
        total_acc += acc

        print('.', end='', flush=True)

    print()
    print('Accuracy: ', (total_acc/my_fmnist.x_val.shape[0]).cpu().numpy(), flush=True)

    # produce the image (almost the same as one iteration of the loop above)
    c0, x0 = my_fmnist.get_val(device)
    mask = (torch.rand_like(x0) < args.noise).float()
    c1 = c_model.sample_random(6)
    z1 = z_model.sample_random(6)
    x1 = x_model.sample(c1, z1) * mask + x0 * (1 - mask)
    xav = torch.zeros_like(x1)

    ops = generate_seq(1000, 3)
    for op in ops:
        if op == 0:
            c1 = c_model.sample(z1, x1)
        elif op == 1:
            z1 = z_model.sample(c1, x1)
        else:
            mu, sigma = x_model.get_ms(c1, z1)
            x1 = torch.randn_like(x1) * sigma + mu
            x1 = x1 * mask + x0 * (1 - mask)
            xav = xav * 0.9 + mu * 0.1

    im1 = x0 * std + mean
    im2 = x0 * (1 - mask) * std + mean
    im3 = (xav * mask + x0 * (1 - mask)) * std + mean
    imtoviz = torch.cat((im1, im2, im3), dim=0)
    vutils.save_image(imtoviz, './images/inp_' + args.id + '_n' + str(args.noise) + '.png', nrow=6)

    print('Done', flush=True)
