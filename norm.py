import torch
from torch import nn


class AdaptiveLayerNorm(nn.Module):
    """AdaLN in DiT"""
    def __init__(self, n_channels, c_dim):
        super().__init__()
        self.fc = nn.Linear(c_dim, 3 * n_channels, bias=True)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, c=None):
        u = torch.mean(x, dim=1, keepdim=True)
        s = (x-u).pow(2).mean(1,keepdim=True)
        x = (x - u) / torch.sqrt(s + 1e-6)
        scale, bias, gamma = torch.chunk(self.fc(c), chunks=3, dim=1)
        scale = scale[:, :, None, None]
        bias = bias[:, :, None, None]
        return x.mul(1 + scale).add(bias), gamma[:, :, None, None]
    


