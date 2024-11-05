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
    
    # Pad the input tensor to handle border cells
    padded_x = F.pad(x, (0, 0, 1, 1, 1, 1, 0, 0)) 
    
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
    
    # Assert that the model's output and the simulated output are close within a tolerance
    print(output_model, output_simulated)
    assert torch.allclose(output_model, output_simulated, atol=1e-6), "NeuralAlgorithm output does not match the simulated output."

