import os
import time
import torch

from inconsistency import inconsistency
from c_model import CModel
from x_model import XModel

if __name__ == '__main__':
    print(os.uname(), flush=True)
    time0 = time.time()

    # printout
    print('# Starting at ' + time.strftime('%c'), flush=True)

    ainc = 0
    ainc2 = 0
    for _ in range(100):
        c_model = CModel(0)
        x_model = XModel(0, False)
        inc = inconsistency(c_model, x_model, 100000, nit_warmup=100, nit_acc=1000)
        ainc += inc
        ainc2 += inc * inc
        print('.', end='', flush=True)

    ainc = ainc / 100
    ainc2 = torch.sqrt(ainc2 / 100 - ainc * ainc)

    print()
    print('Inconsistency:', ainc.numpy(), 'std = ', ainc2.numpy(), flush=True)
