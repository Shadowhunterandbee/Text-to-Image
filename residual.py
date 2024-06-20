import torch
from torch import nn

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Group normalization for the input channels
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)

        # First convolutional layer: convolves input channels to output channels
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Group normalization for the output channels after the first convolution
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)

        # Second convolutional layer: convolves output channels to output channels
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual layer: identity mapping if in_channels == out_channels, otherwise a 1x1 convolution
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # Store the input for residual connection
        residue = x

        # First group normalization and activation
        x = self.groupnorm_1(x)
        x = nn.functional.silu(x)  # Swish activation function

        # First convolutional layer
        x = self.conv_1(x)

        # Second group normalization and activation
        x = self.groupnorm_2(x)
        x = nn.functional.silu(x)  # Swish activation function

        # Second convolutional layer
        x = self.conv_2(x)

        # Apply residual connection
        x = x + self.residual_layer(residue)

        # Return the output of the residual block
        return x
