# MNIST

In order to start the learning invoke, e.g.
```python
python main.py --id 0 --nz 64
```

The script `main.py` accepts the following parameters:

`--id` is a unique identifier (an arbitrary string, mandatory). It is used to address the model as well as its outputs (logs, checkpoints, images). The script creates (if necessary) three subdirectories: `./logs`, `./images` and `./models`, where the learning results are stored in files having `id` as a part of the filenames.

`--nz` is the dimension of the latent space (ineger, mandatory). In our experiments with MNIST we mainly used `nz = 64`.

Further optional parameters are:
| Parameter | Type | Default | Descriprion |
| -------- | ------- | ------- | ------- |
| `--load_id` | string | `<empty>` | Identifier of a pre-trained model to load, empty means "nothing to load". |
| `--stepsize` | float | `1e-4` | Gradient stepsize. |
| `--bs_a` | int | `1024` | The total number of examples in the batch. |
| `--bs_n` | int |  `256` | The number of newly inserted samples. |
| `--niterations` | int | `-1` | The number of gradient update iterations (-1 for infinity).

Note:
* the fraction ` bs_a / bs_n` (i.e. `4` per default) defines the average chain length, see [target.py](./target.py) for details.
* The default `niterations = -1` means that the learning never stops, i.e. should be cancelled manually from outside.
* The learning takes quite a time, so be patient. In our hardware (single GPU NVIDIA H100), 1M update iterations takes about 6 hours. More or less useful results are however obtained much faster (after about 200k iterations).

The progress of the learning can be watched using the notebook `plot.ipynb`. It just reads the logs and the images and visualizes the content.
___

After the learning is done the classification accuracies can be computed using
```python
python inference.py --id 0 --nz 64
```
It just goes through the dataset and computes training and validation accuracies.

## Code structure

The core components of the model are conditional probability distributions, which are implemented in `*_model.py` for classes `c`, latent variables `z` and images `x`. The sources for $p(c\mid z,x)$ in [c_model.py](./c_model.py) are densely commented. Others have a very similar structure. All conditional distributions use neural networks, which are implemented in [nets.py](./nets.py). 

As already mentioned above, in [target.py](./target.py) the blending learning algorithm is implemented.