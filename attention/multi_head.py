import numpy as np
from attention.attention import SelfAttention

class MultiHeadAttention:
    def __init__(self, embedding_dim, num_heads):
        assert embedding_dim % num_heads == 0, f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.Wq = np.random.randn(embedding_dim, embedding_dim)
        self.Wk = np.random.randn(embedding_dim, embedding_dim)
        self.Wv = np.random.randn(embedding_dim, embedding_dim)
        self.Wo = np.random.randn(embedding_dim, embedding_dim)

        self.attention = SelfAttention(self.head_dim)

    def forward(self, x):
        # x is expected to be of shape (batch_size, seq_len, embedding_dim)
        print(x.shape)
        batch_size, seq_len, embed_dim = x.shape

        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        # Reshape and transpose to (batch_size, num_heads, seq_len, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        d = self.head_dim

        # Calculate attention scores
        # Q: (batch_size, num_heads, seq_len, head_dim)
        # K.transpose: (batch_size, num_heads, head_dim, seq_len)
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d)

        # Apply softmax to get attention weights
        weights = self.attention.softmax(scores)

        # Multiply weights with V
        output = weights @ V

        # Concatenate heads
        # Transpose to (batch_size, seq_len, num_heads, head_dim)
        output = output.transpose(0, 2, 1, 3)
        # Reshape to original embedding size: (batch_size, seq_len, embedding_dim)
        output = output.reshape(batch_size, seq_len, embed_dim)

        # Apply output projection
        output = output @ self.Wo

        return output

if __name__ == "__main__":
    # Verification
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 8
    
    # 1. embed_dim is divisible by num_heads (64 % 8 == 0)
    mha = MultiHeadAttention(embedding_dim=embed_dim, num_heads=num_heads)
    
    # 2. Input of shape (batch_size, seq_len, embed_dim)
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    output = mha.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 3. Verify output has the same shape
    assert x.shape == output.shape, f"Shape mismatch! Input: {x.shape}, Output: {output.shape}"
    print("Verification passed: output has the same shape as input!")