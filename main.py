import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import plotly.express as px
from jaxtyping import Float, jaxtyped
from dataclasses import dataclass
from typeguard import typechecked as typechecker
import einops

"""
The model will take in a grid of width x height x channels, and apply the rule learned by
the neural algorithm(which is basically a convolution)

All cells have RGB, alpha(aliveness), and the other channels are for the neural algorithm to use
"""

@dataclass
class EnviromentArguments: # data that describes the "petri dish" that the cells will live in
    width: int
    height: int
    channels: int
    alive_threshold: float


class ClearCells(nn.Module):
    # Will clear values in all cells that don't have any living neighbour (and are dead themselves)
    def __init__(self, args: EnviromentArguments):
        super().__init__()
        self.args = args
        kernel = torch.ones((1, 1, 3, 3))
        self.register_buffer('neighbour_kernel', kernel)
    
    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, 'batch width height channels']):
        alive_cells = x[..., 3] > self.args.alive_threshold
        alive_cells = alive_cells.unsqueeze(1) # batch 1 width height
        alive_cells = alive_cells.float()

        neighbour_sum = F.conv2d(alive_cells, self.neighbour_kernel, padding=1).squeeze(1) # batch width height
        x[neighbour_sum == 0, :] = 0

        return x



