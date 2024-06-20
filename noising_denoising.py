import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

# Define Gaussian noise function for image processing
def add_gaussian_noise(image, mean=0, std=25):
    """
    Adds Gaussian noise to an image.

    Parameters:
    - image: Input image as a NumPy array.
    - mean: Mean (center) of the Gaussian distribution.
    - std: Standard deviation (spread or width) of the distribution.

    Returns:
    - Noisy image as a NumPy array.
    """
    row, col, ch = image.shape
    gauss = np.random.normal(mean, std, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# Self-Attention module
class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        batch_size, sequence_length, d_embed = x.shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, d_embed)
        output = self.out_proj(output)

        return output

# Cross-Attention module
class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        batch_size, seq_len_q, d_embed = x.shape
        seq_len_kv, d_cross = y.shape[1], y.shape[2]
        interim_shape_q = (batch_size, seq_len_q, self.n_heads, self.d_head)
        interim_shape_kv = (batch_size, seq_len_kv, self.n_heads, self.d_head)

        q = self.q_proj(x).view(interim_shape_q).transpose(1, 2)
        k = self.k_proj(y).view(interim_shape_kv).transpose(1, 2)
        v = self.v_proj(y).view(interim_shape_kv).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).reshape(batch_size, seq_len_q, d_embed)
        output = self.out_proj(output)

        return output

# Attention module with Feed-Forward network
class AttentionWithFeedForward(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, ff_dim):
        super().__init__()
        self.self_attention = SelfAttention(n_heads, d_embed)
        self.cross_attention = CrossAttention(n_heads, d_embed, d_cross)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_embed, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_embed)
        )
        self.layer_norm = nn.LayerNorm(d_embed)

    def forward(self, x, y):
        x = self.layer_norm(x + self.self_attention(x))
        x = self.layer_norm(x + self.cross_attention(x, y))
        x = self.layer_norm(x + self.feed_forward(x))
        return x

# Residual Block module
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

# Attention Block module
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

# VAE Encoder
class VAE_Encoder(nn.Module):
    def __init__(self):
        super(VAE_Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x, noise):
        for module in self.encoder:
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        stdev = torch.exp(log_variance * 0.5)
        x = mean + stdev * noise
        x *= 0.18215

        return x

# VAE Decoder
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
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            VAE_ResidualBlock(64, 64),
            VAE_ResidualBlock(64, 64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Ensure output is in [0, 1] range
        )

    def forward(self, x):
        return super().forward(x)


# Example instantiation of the VAE model
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()

    def forward(self, x, noise):
        encoded = self.encoder(x, noise)
        decoded = self.decoder(encoded)
        return decoded

# Example usage:
if __name__ == "__main__":
    # Instantiate the VAE model
    model = VAE()

    # Example usage of adding Gaussian noise to an image
    image = np.random.rand(1, 3, 64, 64)  # Example random image tensor (batch size 1, 3 channels, 64x64)
    noise = torch.randn_like(image)  # Generate noise tensor
    reconstructed_image = model(image, noise)  # Pass through VAE

    print(reconstructed_image.shape)  # Example output shape

