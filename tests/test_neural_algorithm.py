import pytest
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass

# Import the NeuralAlgorithm and EnviromentArguments classes
# Adjust the import path according to your project structure
from main import NeuralAlgorithm, EnviromentArguments

@pytest.fixture
def environment_args():
    return EnviromentArguments(
        width=5,
        height=5,
        channels=16,
        alive_threshold=0.5
    )

def test_neural_algorithm_linear_activation(environment_args):
    # Initialize the linear activation layer
    linear_activation = nn.Linear(9 * environment_args.channels, environment_args.channels)
    
    # For reproducibility, set a fixed seed
    torch.manual_seed(42)
    
    # Initialize weights and bias with known values
    nn.init.uniform_(linear_activation.weight, a=-0.1, b=0.1)
    nn.init.uniform_(linear_activation.bias, a=-0.1, b=0.1)
    
    # Initialize the NeuralAlgorithm model
    model = NeuralAlgorithm(args=environment_args, activation=linear_activation)
    
    # Create a sample input tensor with random values
    batch_size = 2
    x = torch.randn(batch_size, environment_args.width, environment_args.height, environment_args.channels)
    
    # Get the output from the NeuralAlgorithm model
    output_model = model(x)
    
    # Simulate the NeuralAlgorithm using a for-loop
    output_simulated = torch.zeros_like(output_model)
    
    # Extract weights and bias from the linear activation layer
    weight = linear_activation.weight.data  # Shape: (channels, 9 * channels)
    bias = linear_activation.bias.data      # Shape: (channels,)
    
    # Compute alive cells based on the alive_threshold
    alive_cells = x[..., 3] > environment_args.alive_threshold  # Shape: (batch, width, height)
    alive_cells = alive_cells.float().unsqueeze(1)  # Shape: (batch, 1, width, height)
    
    # Define the neighbor kernel (3x3)
    neighbor_kernel = torch.ones((1, 1, 3, 3))
    
    # Compute the sum of alive neighbors using convolution
    neighbor_sum = F.conv2d(alive_cells, neighbor_kernel, padding=1).squeeze(1)  # Shape: (batch, width, height)
    
    # Determine which cells should be cleared (no alive neighbors)
    cleared_cells = neighbor_sum == 0  # Shape: (batch, width, height)
    
    # Pad the input tensor to handle border cells
    padded_x = F.pad(x, (0, 0, 1, 1, 1, 1, 0, 0))  # (pad_left, pad_right, pad_top, pad_bottom)
    
    # Iterate over each element in the batch, width, and height
    for b in range(batch_size):
        for w in range(environment_args.width):
            for h in range(environment_args.height):
                # Extract the 3x3 neighborhood for all channels
                neighborhood = padded_x[b, w:w+3, h:h+3, :]  # Shape: (channels, 3, 3)
                
                # Flatten the neighborhood to a vector of size (9 * channels)
                neighborhood_flat = neighborhood.reshape(-1)  # Shape: (9 * channels,)
                
                # Apply the linear transformation
                updated = torch.matmul(weight, neighborhood_flat) + bias  # Shape: (channels,)
                
                # Assign the computed values to the simulated output tensor
                output_simulated[b, w, h, :] = updated
    
    # Apply the clear cell logic to the simulated output
    # Cells with no alive neighbors should be set to zero
    # Expand cleared_cells to match the channels dimension for broadcasting
    cleared_cells_expanded = cleared_cells.unsqueeze(-1).expand_as(output_simulated)  # Shape: (batch, width, height, channels)
    output_simulated = torch.where(cleared_cells_expanded, torch.zeros_like(output_simulated), output_simulated)
    
    # Assert that the model's output and the simulated output are close within a tolerance
    print("Model Output:\n", output_model)
    print("Simulated Output:\n", output_simulated)
    assert torch.allclose(output_model, output_simulated, atol=1e-6), "NeuralAlgorithm output does not match the simulated output."
