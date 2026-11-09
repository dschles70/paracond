import os
import argparse
import time
import torch

from helpers import vlen, load_model, save_model

from c_model import CModel
from x_model import XModel
from inconsistency import inconsistency
from target import Target

# compute limiting p(c) by sampling
def compute_prior(c_model : CModel,
                  x_model : XModel,
                  nsamples : int = 1000,
                  nit : int = 5):
    c = c_model.sample_random(nsamples)
    for _ in range(nit):
        x = x_model.sample(c)
        c = c_model.sample(x)
    return c.mean()
    
# compute limiting p(x) from p(c) and p(x|c)
def probs(x_model : XModel,
          cprior: float):

    mu0, sigma0 = x_model.get_ms(torch.zeros([1,1]))
    mu1, sigma1 = x_model.get_ms(torch.ones([1,1]))
    x = torch.linspace(-2, 2, 1000)
    p0 = torch.log(sigma0) + ((x - mu0)**2) / (2*(sigma0**2))
    p0 = (1 - cprior) * torch.exp(-p0)
    p1 = torch.log(sigma1) + ((x - mu1)**2) / (2*(sigma1**2))
    p1 = cprior * torch.exp(-p1)
    p = (p0 + p1).flatten()
    return p / p.sum()

# set the encoder weight to the true posterior
# i.e. enforce the consistency
# the weight is set, the bias is unchanged, i.e. learned
# it works only with common_sigma > 0, otherwise it would fail
def set_encoder(c_model : CModel,
                x_model : XModel):

    dmu = x_model.net.mu.weight.data[0,0]
    sigma2 = torch.exp(x_model.net.lsigma * 2)
    c_model.net.lin.weight.data.fill_(dmu / sigma2)

if __name__ == '__main__':
    print(os.uname(), flush=True)
    time0 = time.time()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id',           type=str,   required=True,  help='Identifier.')
    parser.add_argument('--load_id',      type=str,   default='',     help='Identifier to load, empty means "nothing to load".')
    parser.add_argument('--stepsize',     type=float, default=1e-2,   help='Gradient stepsize')
    parser.add_argument('--bs_a',         type=int,   default=10000,  help='Batch size')
    parser.add_argument('--bs_n',         type=int,   default=2500,   help='The number of newly inserted samples')
    parser.add_argument('--common_sigma', type=int,   default=0,      help='When > 0, enforce consistency')
    parser.add_argument('--niterations',  type=int,   default=200000, help='The number of gradient update iterations')

    args = parser.parse_args()

    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    # printout
    logname = './logs/log-' + args.id + '.txt'
    print('# Starting at ' + time.strftime('%c'), file=open(logname, 'w'), flush=True)

    c_model = CModel(args.stepsize)
    x_model = XModel(args.stepsize, (args.common_sigma > 0))

    if args.load_id != '':
        strict_c = True
        strict_x = True
        load_optimizer = False

        loadprot = load_model(c_model, './models/c_' + args.load_id + '.pt', strict=strict_c, load_optimizer=load_optimizer)
        print('z: ', loadprot, flush=True)
        loadprot = load_model(x_model, './models/x_' + args.load_id + '.pt', strict=strict_x, load_optimizer=load_optimizer)
        print('x: ', loadprot, flush=True)

        print('# Models loaded', file=open(logname, 'a'), flush=True)

    target = Target(c_model, x_model, args.bs_a, args.bs_n)

    log_period = 100
    save_period = 1000
    count = 0

    print('# Models prepared, go ...', file=open(logname, 'a'), flush=True)

    loss_c = 0.69 * torch.ones([])
    loss_x = 1.4  * torch.ones([])
    inc = 0

    while True:
        # accumulation speed
        afactor = (1/(count + 1)) if count < 1000 else 0.001
        afactor1 = 1 - afactor

        c, x, l = target.get_batch()
        loss_c = loss_c * afactor1 + c_model.optimize(c, x, l) * afactor
        loss_x = loss_x * afactor1 + x_model.optimize(c, x, l) * afactor
        # set the encoder to be consistent if necessary
        if args.common_sigma > 0:
            set_encoder(c_model, x_model)

        # once in a while print out
        if count % log_period == 0:

            strtoprint = 'time: ' + str(time.time()-time0) + ' count: ' + str(count)

            strtoprint += ' clen: ' + str(vlen(c_model).cpu().numpy())
            strtoprint += ' closs: ' + str(loss_c.cpu().numpy())

            strtoprint += ' xlen: ' + str(vlen(x_model).cpu().numpy())
            strtoprint += ' xloss: ' + str(loss_x.cpu().numpy())

            count1 = count // log_period
            afactor = (1/(count1 + 1)) if count1 < 100 else 0.01
            afactor1 = 1 - afactor

            inc = inc * afactor1 + inconsistency(c_model, x_model, args.bs_a) * afactor

            strtoprint += ' inc: ' + str(inc.numpy())

            print(strtoprint, file=open(logname, 'a'), flush=True)

        # once in a while save the models
        if count % save_period == 0:
            print('# Saving models ...', file=open(logname, 'a'), flush=True)
            save_model(c_model, './models/c_' + args.id + '.pt')
            save_model(x_model, './models/x_' + args.id + '.pt')

            print('# ... done.', file=open(logname, 'a'), flush=True)

        count += 1
        if count == args.niterations:
            break

    print('# Finished at ' + time.strftime('%c') + ', %g seconds elapsed' %
          (time.time()-time0), file=open(logname, 'a'), flush=True)
