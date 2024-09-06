import torch
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        # Linear layers for values, keys, and queries
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Fully connected output layer
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # Layer normalization to stabilize training
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into multiple heads for parallel attention
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Linear projections for values, keys, and queries
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Calculate attention scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)

        # Dropout for regularization
        attention = self.dropout(attention)

        # Get the output by applying the attention weights to values
        out = torch.einsum("nhqk,nvhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        # Pass through the fully connected layer and apply layer normalization
        out = self.fc_out(out)
        out = self.layer_norm(out)

        return out