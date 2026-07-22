from attention.multi_head import MultiHeadAttention
from tokenizer.tokenizer import Tokenizer
from embeddings.embedding import Embedding
import numpy as np
# from attention.attention import SelfAttention

tokenizer = Tokenizer()
tokenizer.load("vocab.json")

sentence = "I love learning AI"

encoded = tokenizer.encode(sentence)

embedding_dim = 8
print("Encoded tokens:")
print(encoded)

embedding_layer = Embedding(len(tokenizer.word2id), embedding_dim)
embeddings = embedding_layer.forward(encoded)

print("embeddings :")
print(embeddings)

# attention = SelfAttention(embedding_dim)
attention = MultiHeadAttention(embedding_dim, num_heads=2)
embeddings = np.expand_dims(embeddings, axis=0)   # (1, 6, 8)

output = attention.forward(embeddings)

print("output:")
print(output)

decoded = tokenizer.decode(encoded)

print("\nDecoded:")
print(decoded)