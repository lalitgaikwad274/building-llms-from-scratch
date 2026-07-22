import numpy as np


class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        
        """
        Initialize embedding matrix.

        vocab_size: Number of words/tokens
        embedding_dim: Size of each embedding vector
        """

        self.weight = np.random.randn(
            vocab_size,
            embedding_dim
        ) * 0.01

    def forward(self, token_ids):

        return self.weight[token_ids]
    

    def get_vector(self, word_id):
        """
        Return embedding vector for one word.
        """
        return self.weight[word_id]

    def similarity(self, id1, id2):
        """
            The similarity method computes

            similarity= (A . B) / (||A|| * ||B||)
                        
            where:

            1 means the vectors point in the same direction (very similar),
            0 means they are orthogonal (unrelated),
            −1 means they point in opposite directions.
        """
        A = self.get_vector(id1)
        B = self.get_vector(id2)

        numerator = np.dot(A, B)
        denominator = np.linalg.norm(A) * np.linalg.norm(B)

        return numerator / denominator


