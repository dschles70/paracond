import os
import argparse
import time
import torch
import torchvision.utils as vutils

from helpers import ensure_dir, load_model

from z_model import ZModel
from x_model import XModel
from my_mnist import MY_MNIST

if __name__ == '__main__':
    print(os.uname(), flush=True)
    time0 = time.time()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id',    type=str, required=True)
    parser.add_argument('--nz',    type=int, required=True)
    parser.add_argument('--train', type=int, required=True)

    args = parser.parse_args()

    os.makedirs('./generated/origin/', exist_ok=True)
    os.makedirs('./generated/lim/',    exist_ok=True)

    bs = 1000

    # printout
    print('# Starting at ' + time.strftime('%c'), flush=True)

    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    # numbers of hidden variables
    nz = args.nz
    nc = 10

    z_model = ZModel(nc, nz, 0).to(device)
    x_model = XModel(nc, nz, 0).to(device)

    loadprot = load_model(z_model, './models/z_' + args.id + '.pt', strict=True, load_optimizer=False)
    print('z: ', loadprot, flush=True)
    loadprot = load_model(x_model, './models/x_' + args.id + '.pt', strict=True, load_optimizer=False)
    print('x: ', loadprot, flush=True)

    my_mnist = MY_MNIST(bs)

    if args.train > 0:
        x_all = torch.split(my_mnist.x_train.to(device), bs)
        c_all = torch.split(my_mnist.c_train.to(device), bs)
    else:
        x_all = torch.split(my_mnist.x_val.to(device), bs)
        c_all = torch.split(my_mnist.c_val.to(device), bs)
    n_batches = len(x_all)

    print(n_batches, 'batches', flush=True)
    for b in range(n_batches):
        n_in_batch = x_all[b].shape[0]

        # save the original images
        for i in range(n_in_batch):
            vutils.save_image(x_all[b][i], './generated/origin/' + ('%06d' % (b*bs + i)) + '.png')

        # use the same c (i.e. perform conditional generation from p(x|c))
        c = c_all[b].to(device)

        # init z randomly
        z = z_model.sample_random(n_in_batch)
        
        # generate
        for _ in range(100):
            x = x_model.sample(c, z)
            z = z_model.sample(c, x)
        
        # take the "decision" for the last sampled z instead of the sample (removes noise a bit)
        with torch.no_grad():
            x = (x_model.net(c, z) > 0).float().detach()

        # save generated images
        for i in range(n_in_batch):
            vutils.save_image(x[i], './generated/lim/' + ('%06d' % (b*bs + i)) + '.png')

        print('.', end='', flush=True)

    print()
    print('done')
