# tests/test_clear_cells.py

import pytest
import torch
from main import ClearCells, EnviromentArguments

@pytest.fixture
def env_args():
    """Fixture to create environment arguments."""
    return EnviromentArguments(width=5, height=5, channels=4, alive_threshold=0.5)

@pytest.fixture
def clear_cells_model(env_args):
    """Fixture to create an instance of ClearCells."""
    return ClearCells(env_args)

def test_clear_cells_no_alive_cells(clear_cells_model, env_args):
    """
    Test that when no cells are alive, the output remains zero.
    """
    # Input tensor: all cells dead
    x = torch.zeros((1, env_args.width, env_args.height, env_args.channels))

    # Apply model
    output = clear_cells_model(x.clone())

    # All cells should remain zero
    assert torch.all(output == 0), "All cells should remain zero when no cells are alive."

def test_clear_cells_alive_cells_with_neighbors(clear_cells_model, env_args):
    """
    Test that alive cells with neighbors are not cleared.
    """
    # Create input where center cell is alive
    x = torch.zeros((1, env_args.width, env_args.height, env_args.channels))
    center = env_args.width // 2, env_args.height // 2
    x[0, center[0], center[1], 3] = 1.0  # Center cell alive

    # Apply model
    output = clear_cells_model(x.clone())

    # The alive cell should remain alive
    assert output[0, center[0], center[1], 3] == 1.0, "Alive cell should remain alive."

def test_clear_cells_dead_cells_with_no_alive_neighbors(clear_cells_model, env_args):
    """
    Test that dead cells with no alive neighbors remain dead (already zero).
    """
    # All cells dead except center
    x = torch.zeros((1, env_args.width, env_args.height, env_args.channels))
    x[0, env_args.width // 2, env_args.height // 2, 3] = 0.4  # Dead cell

    # Apply model
    output = clear_cells_model(x.clone())

    # All cells should remain zero
    assert torch.all(output == 0), "Dead cells with no alive neighbors should remain zero."

def test_clear_cells_dead_cell_with_alive_neighbors(clear_cells_model, env_args):
    """
    Test that dead cells with alive neighbors are not cleared.
    """
    # Cell at (2,2) is dead but has an alive neighbor at (2,1)
    x = torch.zeros((1, env_args.width, env_args.height, env_args.channels))
    x[0, 2, 2, 3] = 0.4  # Dead cell
    x[0, 2, 1, 3] = 0.6  # Alive neighbor

    # Apply model
    output = clear_cells_model(x.clone())

    # The dead cell should remain dead
    assert output[0, 2, 2, 3] == 0.4, "Dead cell with alive neighbors should not be cleared."

def test_clear_cells_alive_cell_with_no_alive_neighbors(clear_cells_model, env_args):
    """
    Test that alive cells with no other alive neighbors remain alive.
    """
    # Alive cell with no alive neighbors
    x = torch.zeros((1, env_args.width, env_args.height, env_args.channels))
    x[0, 2, 2, 3] = 0.6  # Alive cell

    # Apply model
    output = clear_cells_model(x.clone())

    # The alive cell should remain alive
    assert output[0, 2, 2, 3] == 0.6, "Alive cell with no alive neighbors should remain alive."

def test_clear_cells_multiple_cells(clear_cells_model, env_args):
    """
    Test multiple cells with various alive/dead statuses.
    """
    # Setup a 3x3 grid with a mix of alive and dead cells
    x = torch.zeros((1, env_args.width, env_args.height, env_args.channels))
    alive_threshold = env_args.alive_threshold

    # Alive cells
    alive_positions = [(1, 2), (2, 1)]
    for pos in alive_positions:
        x[0, pos[0], pos[1], 3] = 0.6  # Alive

    # Dead cell with alive neighbors
    x[0, 2, 2, 3] = 0.4  # Dead but has alive neighbors

    # Dead cell with no alive neighbors
    x[0, 0, 0, 3] = 0.4  # Dead with no neighbors

    # Apply model
    output = clear_cells_model(x.clone())

    # Check alive cells remain
    for pos in alive_positions:
        assert output[0, pos[0], pos[1], 3] == 0.6, f"Alive cell at {pos} should remain untouched."

    # Dead cell with alive neighbors should remain dead
    assert output[0, 2, 2, 3] == 0.4, "Dead cell with alive neighbors should not be cleared."

    # Dead cell with no alive neighbors should remain zero
    assert output[0, 0, 0, 3] == 0.0, "Dead cell with no alive neighbors should be cleared."
