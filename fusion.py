import torch as th
import torch.nn as nn
import numpy as np

class Fusion(nn.Module):
   
    def __init__(self, n_views, input_sizes):
      
        super().__init__()
        self.n_views = n_views
        self.input_sizes = input_sizes

        flat_sizes = [np.prod(s) for s in input_sizes]
        assert all(s == flat_sizes[0] for s in flat_sizes)
        self.output_size = [flat_sizes[0]]

        self.weights = nn.Parameter(th.full((n_views,), 1.0 / n_views), requires_grad=True)

    def forward(self, inputs):
      
        return _weighted_sum(inputs, self.weights, normalize_weights=True)

    def get_weights(self, softmax=True):
        out = self.weights
        if softmax:
            out = nn.functional.softmax(self.weights, dim=-1)
        return out


def _weighted_sum(tensors, weights, normalize_weights=True):
    
    if normalize_weights:
        weights = nn.functional.softmax(weights, dim=0)
    out = th.sum(weights[None, None, :] * th.stack(tensors, dim=-1), dim=-1)
    return out