# Gaussian Mixture Model

This is the implementation of the Gaussian Mixture Model experiments from Sec. 5.2 of the paper.

We packed the learning of a series of the models in a script for convenience. Hence, to learn the series, just invoke
```bash
script.sh
```
It does not use GPU and takes about half an hour on a consumer laptop. Watch the progress with `plot.ipynb` (actually after the script is done; when watching during the learning, please adapt). After the learning is done, use `plot_probs.ipynb` to see the learned densities, as well as other results, like in Fig. 2 in the paper.
___

The code structure is very similar to others; see, e.g. [mnist](../mnist/) for details. There is, however, something in addition, e.g. explicit computing of the limiting marginal distributions, inconsistency measure, etc. See the comments at the corresponding places in the code.
