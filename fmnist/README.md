# FMNIST

In order to start the learning invoke, e.g.
```python
python main.py --id 0 --nz 256
```

___

After the learning is done, the classification accuracies can be computed using
```python
python inference.py --id 0 --nz 256
```

___

The computation of the FID scores is a bit more complex. First, we generate image sets using

```python
python generate_data.py --id 0 --nz 256 --train 1
```

```python
python -m pytorch_fid --device cuda:0 ./generated/origin/ ./generated/lim/
```
