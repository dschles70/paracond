from c_model import CModel
from x_model import XModel

# This is actually not the inconsistency measure proposed in the paper (sec. 5.2),
# but an "alternative" (not published yet), namely an experimental measure of satisfying 
# the detailed balance constraint (DBC). The inconsistency measure is implemented directly
# in plot_probs.ipynb. This DBC-measure shows similar behaviour and is used only
# for watching during the learning in plot.ipynb.

def inconsistency(c_model : CModel,      # p(c|x)
                  x_model : XModel,      # p(x|c)
                  bs : int = 10000,      # batch size
                  nit_warmup: int = 100, # the number of iteration for the burn-in phase
                  nit_acc : int = 10.    # the number of iteration for the accumulation
                  ):

    # burn-in
    c = c_model.sample_random(bs)
    for _ in range(nit_warmup):
        x = x_model.sample(c)
        c = c_model.sample(x)

    # accumulate
    inc = 0
    for _ in range(nit_acc):
        inc += x_model.inconsistency(c, x)
        x = x_model.sample(c)
        inc += c_model.inconsistency(c, x)
        c = c_model.sample(x)

    return inc / (nit_acc * 2)
