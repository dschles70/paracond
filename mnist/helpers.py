import os
import torch

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def vlen(layer):
    vsum = 0.
    vnn = 0
    for vv in layer.parameters():
        if vv.requires_grad:
            param = vv.data
            vsum = vsum + (param*param).sum()
            vnn = vnn + param.numel()
    return vsum/vnn

def save_model(model : torch.nn.Module, filename: str):
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict()
        }

        torch.save(checkpoint, filename)

def load_model(model : torch.nn.Module, filename : str, strict : bool = True, load_optimizer : bool = True) -> str:
        
        checkpoint = torch.load(filename)
        loadprot_m = model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        if load_optimizer:
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return loadprot_m

def generate_seq(n : int, n_ops : int, last_op : int = -1):
    seq = []
    for _ in range(n):
        ops = torch.randperm(n_ops)
        while ops[0] == last_op:
            ops = torch.randperm(n_ops)
        last_op = ops[-1]
        seq += ops.tolist()

    return seq
