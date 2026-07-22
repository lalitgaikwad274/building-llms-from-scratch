from embedding import Embedding

embedding = Embedding(
    vocab_size=10,
    embedding_dim=4
)

tokens = [2, 5, 1]

vectors = embedding.forward(tokens)

print(vectors)

similarity = embedding.similarity(2, 5)
