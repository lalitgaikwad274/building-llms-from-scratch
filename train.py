from tokenizer import Tokenizer

with open("dataset.txt") as f:
    text = f.read()

tokenizer = Tokenizer()

tokenizer.fit(text)

tokenizer.save("vocab.json")

print("Vocabulary Size:", tokenizer.word2id)