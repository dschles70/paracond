import os
import argparse
import time
import torch
import torchvision.utils as vutils

from helpers import vlen, load_model, save_model

from c_model import CModel
from z_model import ZModel
from x_model import XModel
from target import Target
from inference import inference

if __name__ == '__main__':
    print(os.uname(), flush=True)
    time0 = time.time()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id',          type=str,   required=True, help='Identifier')
    parser.add_argument('--nz',          type=int,   required=True, help='The number of latent dimensions')
    parser.add_argument('--load_id',     type=str,   default='',    help='Identifier to load, empty means "nothing to load"')
    parser.add_argument('--stepsize',    type=float, default=1e-4,  help='Gradient stepsize')
    parser.add_argument('--bs_a',        type=int,   default=1024,  help='Batch size')
    parser.add_argument('--bs_n',        type=int,   default=256,   help='The number of newly inserted samples')
    parser.add_argument('--niterations', type=int,   default=-1,    help='The number of gradient update iterations (-1 for infinity)')

    # note that the defaults values of bs_a and bs_n correspond to the average chain length equal 4

    args = parser.parse_args()

    # create if necessary the folders containing the learning results -- logs, model checkpoints, images
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./images', exist_ok=True)

    # printout
    logname = './logs/log-' + args.id + '.txt'
    print('# Starting at ' + time.strftime('%c'), file=open(logname, 'w'), flush=True)

    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    # numbers of hidden variables
    nz = args.nz

    # 10 classes in MNIST
    nc = 10

    # create models
    c_model = CModel(nc, nz, args.stepsize).to(device)
    z_model = ZModel(nc, nz, args.stepsize).to(device)
    x_model = XModel(nc, nz, args.stepsize).to(device)

    # load models if necessary
    if args.load_id != '':
        strict_c = True
        strict_z = True
        strict_x = True
        load_optimizer = False

        loadprot = load_model(c_model, './models/c_' + args.load_id + '.pt', strict=strict_c, load_optimizer=load_optimizer)
        print('c: ', loadprot, flush=True)
        loadprot = load_model(z_model, './models/z_' + args.load_id + '.pt', strict=strict_z, load_optimizer=load_optimizer)
        print('z: ', loadprot, flush=True)
        loadprot = load_model(x_model, './models/x_' + args.load_id + '.pt', strict=strict_x, load_optimizer=load_optimizer)
        print('x: ', loadprot, flush=True)

        print('# Models loaded', file=open(logname, 'a'), flush=True)

    # create "target" providing re-sampling functionality for "nlanding" learning
    target = Target(c_model, z_model, x_model, args.bs_a, args.bs_n, device)

    # how often to printout, save the checkpoints
    log_period = 100
    save_period = 1000
    count = 0

    print('# Models prepared, go ...', file=open(logname, 'a'), flush=True)

    # losses to watch
    loss_c = 2.3 * torch.ones([], device=device) # ~ln(10)
    loss_z = 0.69 * torch.ones([], device=device) # ~ln(2)
    loss_x = 0.69 * torch.ones([], device=device) # ~ln(2)

    # accuracies to watch
    acc_train = 0.1 * torch.ones([], device=device) # 1/10 (random decision)
    acc_val = 0.1 * torch.ones([], device=device) # 1/10 (random decision)

    while True:
        # accumulation speed
        afactor = (1/(count + 1)) if count < 1000 else 0.001
        afactor1 = 1 - afactor

        # implements "blending" learning
        # the "target" module prepares the batch, in which the samples are distributed
        # along the chain
        c, z, x, l = target.get_batch()

        # all conditionals are learned on the while batch
        loss_c = loss_c*afactor1 + c_model.optimize(c, z, x, l)*afactor
        loss_z = loss_z*afactor1 + z_model.optimize(c, z, x, l)*afactor
        loss_x = loss_x*afactor1 + x_model.optimize(c, z, x, l)*afactor

        # once awhile print out (in the log-file)
        if count % log_period == 0:

            count1 = count // log_period
            afactor = (1/(count1 + 1)) if count1 < 100 else 0.01
            afactor1 = 1 - afactor

            strtoprint = 'time: ' + str(time.time()-time0) + ' count: ' + str(count)

            strtoprint += ' clen: ' + str(vlen(c_model).cpu().numpy())
            strtoprint += ' closs: ' + str(loss_c.cpu().numpy())

            # inference (train)
            ci, xi = target.my_mnist.get_train(device)
            dec = inference(xi, c_model, z_model)
            ci = torch.argmax(ci, dim=1)
            dec = torch.argmax(dec, dim=1)
            acc = (ci == dec).float().mean()
            acc_train = acc_train * afactor1 + acc * afactor

            # inference (val)
            ci, xi = target.my_mnist.get_val(device)
            dec = inference(xi, c_model, z_model)
            ci = torch.argmax(ci, dim=1)
            dec = torch.argmax(dec, dim=1)
            acc = (ci == dec).float().mean()
            acc_val = acc_val * afactor1 + acc * afactor

            strtoprint += ' cacctrain: ' + str(acc_train.cpu().numpy())
            strtoprint += ' caccval: ' + str(acc_val.cpu().numpy())

            strtoprint += ' zlen: ' + str(vlen(z_model).cpu().numpy())
            strtoprint += ' zloss: ' + str(loss_z.cpu().numpy())

            strtoprint += ' xlen: ' + str(vlen(x_model).cpu().numpy())
            strtoprint += ' xloss: ' + str(loss_x.cpu().numpy())

            print(strtoprint, file=open(logname, 'a'), flush=True)

        # once awhile save the models, paint an image
        if count % save_period == 0:
            print('# Saving models ...', file=open(logname, 'a'), flush=True)
            save_model(c_model, './models/c_' + args.id + '.pt')
            save_model(z_model, './models/z_' + args.id + '.pt')
            save_model(x_model, './models/x_' + args.id + '.pt')

            # sample from the limiting distribution p(z,x|c), i.e. conditioned on c
            c_sta = torch.eye(nc, device=device)
            z_sta = z_model.sample_random(10)
            for _ in range(100):
                x_sta = x_model.sample(c_sta, z_sta)
                z_sta = z_model.sample(c_sta, x_sta)

            with torch.no_grad():
                im1 = x[0:10] # original images
                im2 = x_model.get_probs(c[0:10], z[0:10]) # "reconstructed" images

                # images sampled from random codes (conditioned on the given classes)
                z_tmp = z_model.sample_random(10)
                im3 = x_model.get_probs(c_sta, z_tmp)

                im4 = x_model.get_probs(c_sta, z_sta) # sampled from the limiting distribution
                imtoviz = torch.cat((im1, im2, im3, im4), dim=0)
                vutils.save_image(imtoviz, './images/img_' + args.id + '.png', nrow=10)

            print('# ... done.', file=open(logname, 'a'), flush=True)

        count += 1
        if count == args.niterations:
            break

    print('# Finished at ' + time.strftime('%c') + ', %g seconds elapsed' %
          (time.time()-time0), file=open(logname, 'a'), flush=True)
