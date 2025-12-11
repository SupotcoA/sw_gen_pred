import torch
import math


@torch.no_grad()
def naive_baseline(z, x0, **kwargs):
    b,s,d=x0.shape
    M_per_loop = z.shape[0]//b

    sample_mean = x0[:, :, ::4]
    