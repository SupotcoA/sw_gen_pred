import torch
from torch import nn
from norm import AdaptiveLayerNorm
from diffloss_mar import SimpleMLPAdaLN

class MLP(nn.Module):
    """MLP for transformer block"""
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity=nn.GELU, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nonlinearity(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
