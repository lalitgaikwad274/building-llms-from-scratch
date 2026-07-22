import numpy as np
from attention import SelfAttention
# Use absolute import to resolve embedding module
# from ..embeddings import Embedding
# from embeddings.embedding import Embedding


# embedding = Embedding(
#     vocab_size=10,
#     embedding_dim=4
# )
embeddings = np.random.randn(3, 4)

attention = SelfAttention(4)

scores, V = attention.forward(embeddings)

weights = attention.softmax(scores)

print("scores")
print(scores)

print("V")
print(V)

print(weights)