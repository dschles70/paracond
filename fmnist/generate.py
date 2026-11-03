import os
import argparse
import time
import torch
import torchvision.utils as vutils

from helpers import load_model

from z_model import ZModel
from x_model import XModel

if __name__ == '__main__':
    print(os.uname(), flush=True)
    time0 = time.time()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id',    type=str, required=True)
    parser.add_argument('--nz',    type=int, required=True)

    args = parser.parse_args()

    os.makedirs('./images/', exist_ok=True)

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

    # use the same c (i.e. perform conditional generation from p(x|c))
    c = [torch.eye(10)] * 4
    c = torch.cat(c).to(device)

    # init z randomly
    z = z_model.sample_random(40)
        
    # generate
    for _ in range(1000):
        x = x_model.sample(c, z)
        z = z_model.sample(c, x)
        
    # take the mean values for the last sampled z (denormalized)
    x = x_model.get_ms(c, z)[0] * 0.353 + 0.286

    # save the generated image
    vutils.save_image(x, './images/generated_' + args.id + '.png', nrow=10)

    print('done')
