# Paracond

This repository contains implementations accomplishing:

> Dmitrij Schlesinger, Boris Flach, and Alexander Shekhovtsov. _Deep multivariate models with parametric conditionals_, 2026. [https://arxiv.org/abs/2602.01953](https://arxiv.org/abs/2602.01953).

The code here is "younger" than the one used in the paper. After it was published, there were some minor changes: slightly different network architectures, some stability improvements, etc. Hence, we cannot guarantee full reproducibility. The new code presented here is just slightly better, producing, however, very similar results with respect to used metrics like, e.g. classification accuracy, FID scores, etc.

The subfolders contain implementations for different datasets. Thereby, [`mnist`](./mnist) contains a detailed description of how to use the code, and the sources are exhaustively commented. Others — i.e. `fmnist` and `celeba` — have a very similar structure. They are documented rather sparsely where it is necessary.

_Currently, there is only [`mnist`](./mnist) included; others follow._
