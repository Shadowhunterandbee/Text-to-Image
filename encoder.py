import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock
from attention import SelfAttention

# VAE Variational Auto Encoder
def compute_stdev(log_variance):
    # Clamp the log variance to prevent numerical instability
    log_variance = torch.clamp(log_variance, -30, 20)

    # Compute standard deviation from log variance
    stdev = torch.exp(log_variance * 0.5)
    return stdev


def compute_variance(log_variance):
    # Clamp the log variance to prevent numerical instability
    log_variance = torch.clamp(log_variance, -30, 20)

    # Compute variance from log variance
    variance = torch.exp(log_variance)
    return variance


def kl_divergence(mean, log_variance):
    # Compute KL divergence between the learned distribution and standard normal distribution
    kl_div = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp(), dim=1)
    return kl_div.mean()  # Return the mean KL divergence across the batch


def sample_from_normal(mean, stdev):
    # Reparameterization trick: sample latent variable z
    epsilon = torch.randn_like(stdev)
    z = mean + stdev * epsilon
    return z


def reconstruction_loss(x, x_recon):
    # Compute binary cross-entropy (BCE) loss for reconstruction
    bce_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    return bce_loss

#
# # Example usage of the functions:
# if __name__ == "__main__":
#     # Initialize encoder and decoder
#     encoder = VAE_Encoder()
#     decoder = VAE_Decoder()
#
#     # Example forward pass through VAE
#     input_image = torch.randn(1, 3, 64, 64)  # Example input image
#     noise = torch.randn(1, 8, 16, 16)  # Example noise for sampling latent variable
#
#     # Forward pass through encoder
#     mean_logvar = encoder(input_image, noise)
#
#     # Split mean and log variance
#     mean, log_variance = torch.chunk(mean_logvar, 2, dim=1)
#
#     # Compute standard deviation
#     stdev = compute_stdev(log_variance)
#
#     # Sample latent variable z
#     z = sample_from_normal(mean, stdev)
#
#     # Forward pass through decoder
#     reconstructed_image = decoder(z)
#
#     # Compute reconstruction loss
#     loss_recon = reconstruction_loss(input_image, reconstructed_image)
#
#     print("Reconstruction Loss:", loss_recon.item())

def compute_stdev(self, log_variance):
    # Clamp the log variance to prevent numerical instability
    log_variance = torch.clamp(log_variance, -30, 20)

    # Compute standard deviation from log variance
    stdev = torch.exp(log_variance * 0.5)
    return stdev


def compute_variance(self, log_variance):
    # Clamp the log variance to prevent numerical instability
    log_variance = torch.clamp(log_variance, -30, 20)

    # Compute variance from log variance
    variance = torch.exp(log_variance)
    return variance

class VAE_Encoder(nn.Module):
    def __init__(self):
        super(VAE_Encoder, self).__init__()

        # Define the encoder network architecture
        self.encoder = nn.Sequential(
            # Initial convolution layer to process input images
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # Residual blocks for feature extraction
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            # Downsampling layer to reduce spatial dimensions
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # More residual blocks at a higher feature dimension
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            # Further downsampling to reduce spatial dimensions
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # Residual blocks at a larger feature dimension
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            # Final downsampling to compress spatial dimensions
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # More residual blocks for deep feature extraction
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # Attention block to focus on important features
            VAE_AttentionBlock(512),

            # Group normalization for improved gradient flow
            nn.GroupNorm(32, 512),

            # Scaled Exponential Linear Unit (SiLU) activation function
            nn.SiLU(),

            # Convolution layer to reduce feature dimensionality
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # Additional 1x1 convolution for final feature adjustment
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )


    def forward(self, x, noise):
        # Apply each module in the encoder sequentially
        for module in self.encoder:
            # Handle asymmetric padding for downsampling layers
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))  # Add padding to maintain dimensions
            x = module(x)  # Apply the current module to input x

        # Split the output into mean and log variance components
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # Clamp the log variance to prevent numerical instability
        log_variance = torch.clamp(log_variance, -30, 20)

        # Compute the standard deviation from the log variance
        stdev = torch.exp(log_variance * 0.5)

        # Reparameterization trick: sample latent variable z
        x = mean + stdev * noise

        # Scale the output by a constant factor for stability
        x *= 0.18215

        return x
# Start with the reverse process i.e decoder