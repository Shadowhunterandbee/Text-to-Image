import torch
from torch import nn
from torch.nn import functional as F
import torch
from torch import nn
from torch.nn import functional as F
# CLIPEmbedding module definition

class BertStyleEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_position_embeddings=512):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_size)
        self.token_type_embeddings = nn.Embedding(2, embedding_size)  # For segment embeddings

        # Layer normalization for each embedding type
        self.LayerNorm = nn.LayerNorm(embedding_size, eps=1e-12)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)

        # Get token embeddings for input_ids
        token_embeddings = self.token_embeddings(input_ids)

        # Create position embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Add them together
        embeddings = token_embeddings + position_embeddings

        # Add segment embeddings if provided
        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        # Layer normalization
        embeddings = self.LayerNorm(embeddings)

        return embeddings

# Example usage:
if __name__ == "__main__":
    vocab_size = 20000  # Example vocabulary size
    embedding_size = 768  # Example embedding size
    max_position_embeddings = 512  # Maximum sequence length
    batch_size = 32
    seq_length = 128

    # Example input_ids and token_type_ids
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    token_type_ids = torch.randint(0, 2, (batch_size, seq_length))

    # Create BERT-style embedding instance
    bert_embedding = BertStyleEmbedding(vocab_size, embedding_size, max_position_embeddings)

    # Forward pass
    embeddings = bert_embedding(input_ids, token_type_ids)
    print(f"Output shape: {embeddings.shape}")  # Expected: (batch_size, seq_length, embedding_size)

from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        residue = x

        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)
        x = self.linear_2(x)
        x += residue

        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([CLIPLayer(12, 768) for i in range(12)])
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)

        return output

if __name__ == "__main__":
    tokens = torch.randint(0, 49408, (32, 77))  # Batch_Size x Seq_Len
    model = CLIP()
    output = model(tokens)
    print(f"Output shape: {output.shape}")
