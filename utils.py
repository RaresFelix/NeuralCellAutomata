from PIL import Image
from typing import Tuple, List
import os
from jaxtyping import Float, jaxtyped
from torch import Tensor
import numpy as np
import einops
import plotly.graph_objects as go
#import px
from plotly import express as px
from typeguard import typechecked as typechecker

def load_image_and_downscale(file_name: str, target_size: Tuple[int, int]):
    DEBUG = False
    
    full_path = os.path.join('/workspace/NeuralCellAutomata/images', file_name)
    image = Image.open(full_path)
    image = image.resize(target_size, Image.Resampling.NEAREST)
    if DEBUG:
        px.imshow(image).show()
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    return image

@jaxtyped(typechecker=typechecker)
def render_grid(grid: Float[Tensor, 'width height channels'], title: str = 'Grid Visualization'):
    grid = grid[..., :4]  # Only keep RGB alpha channels
    if grid.shape[-1] < 4:
        raise ValueError("Grid must have at least 4 channels (RGB + Alpha).")
    
    rgb = grid[..., :3].cpu().clone().detach().numpy()
    alpha = grid[..., 3].cpu().clone().detach().numpy()

    rgb = einops.rearrange(rgb, 'w h c -> h w c')
    alpha = einops.rearrange(alpha, 'w h -> h w')

    background = np.ones_like(rgb)  # White background
    alpha_expanded = np.expand_dims(alpha, axis=-1)  # Shape: (height, width, 1)
    blended = rgb * alpha_expanded + background * (1 - alpha_expanded)

    # Ensure the values are within [0, 1]
    blended = np.clip(blended, 0, 1)

    fig = go.Figure(data=go.Image(z=blended))
    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    fig.show()


def animate_grids(grids: List[Float[Tensor, 'width height channels']], 
                 title: str = 'Grid Animation',
                 frame_duration: int = 100):
    """
    Animates a list of grids (tensors) using Plotly.

    Parameters:
    - grids: List of tensors, each of shape [width, height, channels]
    - title: Title of the animation
    - frame_duration: Duration of each frame in milliseconds
    """
    if not grids:
        raise ValueError("The list of grids is empty.")

    processed_images = []
    for idx, grid in enumerate(grids):
        grid = grid[..., :4]  # Only keep RGB alpha channels
        if grid.shape[-1] < 4:
            raise ValueError(f"Grid at index {idx} must have at least 4 channels (RGB + Alpha).")
        
        rgb = grid[..., :3].cpu().clone().detach().numpy()
        alpha = grid[..., 3].cpu().clone().detach().numpy()

        rgb = einops.rearrange(rgb, 'w h c -> h w c')
        alpha = einops.rearrange(alpha, 'w h -> h w')

        background = np.ones_like(rgb)  # White background
        alpha_expanded = np.expand_dims(alpha, axis=-1)  # Shape: (height, width, 1)
        blended = rgb * alpha_expanded + background * (1 - alpha_expanded)

        # Ensure the values are within [0, 1]
        blended = np.clip(blended, 0, 1)

        # Convert to 0-255 and uint8 for Plotly
        blended_uint8 = (blended * 255).astype(np.uint8)
        processed_images.append(blended_uint8)

    # Create initial figure
    fig = go.Figure()

    # Add initial frame
    fig.add_trace(go.Image(z=processed_images[0]))

    # Create frames
    frames = [go.Frame(data=[go.Image(z=img)], name=str(i)) for i, img in enumerate(processed_images)]

    # Add frames to figure
    fig.frames = frames

    # Define animation settings
    fig.update_layout(
        title=title,
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": frame_duration, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ],
        xaxis=dict(scaleanchor="y", scaleratio=1, visible=False),
        yaxis=dict(scaleanchor="x", scaleratio=1, visible=False),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    # Set frame layout
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.show()
