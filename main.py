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
    """
        Will clear values in all cells that don't have any living neighbour in a 3 x 3 (and are dead themselves)
    """
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

@jaxtyped(typechecker=typechecker)
class NeuralAlgorithm(nn.Module):
    def __init__(self, args: EnviromentArguments, activation: nn.Module):
        super().__init__()
        self.args = args
        self.activation = activation  # activation should take 9 * args.channels inputs and output args.channels
        kernel = torch.ones((1, 1, 3, 3))
        self.register_buffer('neighbour_kernel', kernel)

    def forward(self, x: Float[Tensor, 'batch width height channels']) -> Float[Tensor, 'batch width height channels']:
        alive_cells = x[..., 3] > self.args.alive_threshold
        alive_cells = alive_cells.unsqueeze(1) # batch 1 width height
        alive_cells = alive_cells.float()
        neighbour_sum = F.conv2d(alive_cells, self.neighbour_kernel, padding=1).squeeze(1) # batch width height
        cleared_cells = neighbour_sum == 0

        padded_x = F.pad(x, (0, 0, 1, 1, 1, 1, 0, 0))  # (pad_left, pad_right, pad_top, pad_bottom)
        
        # Extract 3x3 neighborhoods
        neighbors = torch.cat([
            padded_x[:, i:i+self.args.height, j:j+self.args.width, :]
            for i in range(3) for j in range(3)
        ], dim=-1)  # Concatenate along the channel dimension

        print(neighbors.shape)
        
        # Apply activation
        updated_state = self.activation(neighbors)
        
        # Reshape back to original tensor shape
        updated_state = updated_state.view(x.shape)

        updated_state[cleared_cells, :] = 0
        return updated_state