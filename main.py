import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import plotly.express as px
from jaxtyping import Float, jaxtyped
from dataclasses import dataclass
from typeguard import typechecked as typechecker
import einops
from utils import load_image_and_downscale, render_grid, animate_grids

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
        kernel = torch.ones((1, 1, 3, 3), device=device)
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
    def __init__(self, args: EnviromentArguments,
            activation: nn.Module,
            activation_args: dict = None,
            device:torch.device = torch.device('cuda')
        ):
        super().__init__()
        # activation should take 9 * args.channels inputs and output args.channels
        self.args = args
        self.activation = activation(**activation_args)
        kernel = torch.ones((1, 1, 3, 3), device=device)
        self.register_buffer('neighbour_kernel', kernel)

        with torch.no_grad():
            test_input = torch.zeros((1, args.height, args.width, 9 * args.channels))
            try:
                test_output = self.activation(test_input)
            except Exception as e:
                print(f"Activation function failed on sample testcase(may be due to dimension errors) with error: {e}")
                raise

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
        
        # Apply activation
        updated_state = self.activation(neighbors)
        
        # Reshape back to original tensor shape
        updated_state = updated_state.view(x.shape)

        updated_state[cleared_cells, :] = 0

        x = x + updated_state
        #clip 0 to 1 each
        x = torch.clamp(x, -1, 1)
        return x

class MLP(nn.Module):
    def __init__(self, args: EnviromentArguments, hidden_size: int):
        super().__init__()
        self.args = args
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(9 * args.channels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, args.channels)
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x: Float[Tensor, 'batch width height 9 * channels']) -> Float[Tensor, 'batch width height channels']:
        x = x.view(-1, 9 * self.args.channels)
        x = F.tanh(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    #squre grid for now
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    args = EnviromentArguments(width=20, height=20, channels=16, alive_threshold=0.1)
    target_image = torch.tensor(load_image_and_downscale('smiling_emoji.png', (10, 10))).to(device)

    # Init grid as black with a white pixel in the middle
    starting_grid = torch.zeros((1, args.height, args.width, args.channels))
    starting_grid[0, args.height // 2, args.width // 2, :] = .8  # White pixel


    # target grid has target image in the middle
    target_grid = torch.zeros((1, args.height, args.width, args.channels)).to(device)
    for i in range(target_image.shape[0]):
        for j in range(target_image.shape[1]):
            target_grid[0, i + args.height // 2 - target_image.shape[0] // 2, j + args.width // 2 - target_image.shape[1] // 2, :4] = target_image[i][j]

    model = NeuralAlgorithm(args, MLP, dict(args=args, hidden_size = 64)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    diffusion_time = 16
    batch_count = 64

    states = []
    cnt_step = 200
    for step in range(cnt_step):
        grid = starting_grid.detach().clone().to(device)
        grid = grid.repeat(batch_count, 1, 1, 1)
        
        ema_loss = torch.tensor(0.0).to(device)
        ema_loss.requires_grad = True

        for t in range(diffusion_time):
            if step == cnt_step - 1:
                states.append(grid.clone()[0])
            grid = model(grid)
            #add slight noise
            grid += torch.randn_like(grid) * 0.001
            loss = F.mse_loss(grid[..., :3], target_grid[..., :3])
            ema_loss = ema_loss * 0.9 + loss * 0.1
        if step == cnt_step - 1:
            states.append(grid.clone()[0])
        if step % 50 == 0:
            print(f"Step {step}, loss: {ema_loss.item():.4f}")

        optimizer.zero_grad()
        ema_loss.backward()
        optimizer.step()
    animate_grids(states, title="Optimization Progress")
    pass

if __name__ == '__main__':
    main()