import numpy as np


class SelfAttention:
    def __init__(self, embedding_dim):
        self.Wq = np.random.randn(
            embedding_dim,
            embedding_dim
        )

        self.Wk = np.random.randn(
            embedding_dim,
            embedding_dim
        )

        self.Wv = np.random.randn(
            embedding_dim,
            embedding_dim
        )

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        exp = np.exp(x)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def forward(self, x):

        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        d = K.shape[-1]

        scores = (Q @ K.T) / np.sqrt(d)

        # Attention weights
        weights = self.softmax(scores)

        output = weights @ V
        return output 
    
 