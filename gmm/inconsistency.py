from c_model import CModel
from x_model import XModel

def inconsistency(c_model : CModel,
                  x_model : XModel,
                  bs : int = 10000,
                  nit_warmup: int = 100,
                  nit_acc : int = 10):


    c = c_model.sample_random(bs)
    for _ in range(nit_warmup):
        x = x_model.sample(c)
        c = c_model.sample(x)

    inc = 0
    for _ in range(nit_acc):
        inc += x_model.inconsistency(c, x)
        x = x_model.sample(c)
        inc += c_model.inconsistency(c, x)
        c = c_model.sample(x)

    return inc / (nit_acc * 2)
