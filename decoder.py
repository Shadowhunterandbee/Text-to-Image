import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()

        # Define convolutional layers for spatial attention
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)  # Reduce to 1 channel
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute spatial attention map
        att_map = self.conv1(x)
        att_map = self.sigmoid(att_map)

        # Apply the attention map to the input
        out = x * att_map

        return out, att_map

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super(RelativePositionalEncoding, self).__init__()

        # Maximum sequence length and embedding dimension
        self.max_len = max_len
        self.embed_dim = embed_dim

        # Embedding table for relative positional encodings
        self.relative_positions = nn.Embedding(2 * max_len - 1, embed_dim)

    def forward(self, positions):
        """
        positions: Tensor of shape (seq_len, seq_len) representing relative positions.
                   positions[i, j] should be in the range [-max_len+1, max_len-1].
        """
        # Ensure positions are in the correct range
        positions = torch.clip(positions, -self.max_len + 1, self.max_len - 1) + self.max_len - 1

        # Lookup relative positional embeddings
        embeddings = self.relative_positions(positions)

        return embeddings

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        residue = x
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        x += residue
        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

        self.initialize_weights()  # Initialize weights during instantiation

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='silu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x /= 0.18215  # Scaling factor from the Encoder
        for module in self:
            x = module(x)
        return x


# Example usage:
decoder = VAE_Decoder()
input_tensor = torch.randn(8, 4, 32, 32)  # Example input tensor (Batch_Size=8, 4 channels, 32x32 size)
output_tensor = decoder(input_tensor)
print("Output shape:", output_tensor.shape)
print("Number of parameters in the decoder:", decoder.get_number_of_parameters())
