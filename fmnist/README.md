# Fashion MNIST

The model for Fashion MNIST is exactly the same as for MNIST (see [mnist](../mnist/)) except that we use grey-valued images $x\in\mathbb{R}^{28\times 28}$ and model $p(x\mid c,z)$ as a Gaussian $p(x\mid c,z)=\mathcal N(x;\mu(c,z),\sigma)$, where the means $\mu(c,z)$ for each pixel are provided by a feedforward network and $\sigma\in\mathbb R$ is a global learnable parameter (see [x_model.py](./x_model.py)). Correspondingly, there are also some minor changes at several other places, e.g. the dataset wrapper or visualizations. In addition, we increased the dimension of the latent space from $m=64$ to $m=256$.

Below are the example calls:

Learning:
```python
python main.py --id 0 --nz 256
```
(takes about 8.5 hours on NVIDIA H100 for 1M update iterations).

Conditional generation:
```python
python generate.py --id 0 --nz 256
```
The resulting image is stored in `./images/generated_<id>.png`.

Classification accuracies:
```python
python inference.py --id 0 --nz 256
```
This implementation achieves `97.89`% training accuracy and `89.57`% validation accuracy.

The computation of the FID scores:

```python
python generate_data.py --id 0 --nz 256 --train 1
python -m pytorch_fid --device cuda:0 ./generated/origin/ ./generated/lim/
```
(repeat with `--train 0` for validation, do not forget to delete the subfolder `./generated`). This implementation gives the training FID score `31.14` and the validation FID score `33.54`.
___

There is an additional experiment for Fashion MNIST, namely the classification from incomplete inputs combined with inpainting (will be documented later). In order to perform it, run e.g.
```python
python infinp.py --id 0 --nz 256 --noise 0.5
```
This implementation reports the accuracy of `85.57`% (for the noise level `0.5`) and produces the inpainted image stored in `./images/inp_<id>_n<noise>.png`, see [infinp.py](./infinp.py).
