import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for Transformer networks.

    Args:
        n_heads : Number of attention heads.
        d_embed : Dimension of input embeddings.
        in_proj_bias: Whether to include bias in input projection layers.
        out_proj_bias: Whether to include bias in output projection layer.
    """

    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        # Projecting input embeddings into query, key, and value spaces
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)

        # Projecting output from attention heads back to original space
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        """
        Forward pass of the self-attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Seq_Len, Dim).
            causal_mask (bool): Whether to apply a causal mask to prevent attending to future positions.

        Returns:
            torch.Tensor: Output tensor of shape (Batch_Size, Seq_Len, Dim).
        """

        # Getting the shape of the input tensor
        input_shape = x.shape

        # Retrieving batch size, sequence length, and embedding dimension
        batch_size, sequence_length, d_embed = input_shape

        # Creating intermediate shape for multi-head attention computation
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # Projecting input tensor into query, key, and value tensors
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # Reshaping and transposing to prepare for multi-head attention computation
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Calculating attention scores
        weight = q @ k.transpose(-1, -2)

        # Applying causal mask if specified
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        # Scaling attention scores and applying softmax
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # Applying attention to value tensor
        output = weight @ v

        # Rearranging output tensor dimensions
        output = output.transpose(1, 2).reshape(input_shape)

        # Projecting output tensor back to original dimension
        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for Transformer networks.

    Args:
        n_heads (int): Number of attention heads.
        d_embed (int): Dimension of input embeddings (query).
        d_cross (int): Dimension of input embeddings (key and value).
        in_proj_bias (bool): Whether to include bias in input projection layers.
        out_proj_bias (bool): Whether to include bias in output projection layer.
    """

    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        # Projecting query, key, and value spaces
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)

        # Projecting output from attention heads back to original space
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        """
        Forward pass of the cross-attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Seq_Len_Q, Dim_Q) representing query.
            y (torch.Tensor): Input tensor of shape (Batch_Size, Seq_Len_KV, Dim_KV) representing key and value.

        Returns:
            torch.Tensor: Output tensor of shape (Batch_Size, Seq_Len_Q, Dim_Q).
        """

        # Getting the shape of the query tensor
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        # Projecting query, key, and value tensors
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # Creating intermediate shape for multi-head attention computation
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # Reshaping and transposing to prepare for multi-head attention computation
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Calculating attention scores
        weight = q @ k.transpose(-1, -2)

        # Scaling attention scores and applying softmax
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # Applying attention to value tensor
        output = weight @ v

        # Rearranging output tensor dimensions
        output = output.transpose(1, 2).reshape(input_shape)

        # Projecting output tensor back to original dimension
        output = self.out_proj(output)

        return output
