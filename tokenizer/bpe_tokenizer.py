import json
from collections import Counter

class BPETokenizer:
    def __init__(self):
        self.merges = []

    # -----------------------------
    # Build initial vocabulary
    # -----------------------------

    def get_vocab(self, corpus):
        vocab = Counter()

        for word in corpus:
            vocab[" ".join(list(word)) + " </w>"] += 1

        print("------- get vocab ------")
        print(vocab)
        return vocab


    # -----------------------------
    # Count adjacent symbol pairs
    # -----------------------------
    def get_stats(self, vocab):
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    # -----------------------------
    # Merge best pair
    # -----------------------------
    def merge_vocab(self, pair, vocab):
        new_vocab = {}

        bigram = " ".join(pair)
        replacement = "".join(pair)

        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]

        return new_vocab

    # -----------------------------
    # Train BPE
    # -----------------------------
    def train(self, corpus, num_merges=20):

        vocab = self.get_vocab(corpus)

        for i in range(num_merges):
            pairs = self.get_stats(vocab)

            print("======== pairs ==========")
            print(pairs)

            if not pairs:
                break

            best = max(pairs, key=pairs.get)

            self.merges.append(best)
            vocab = self.merge_vocab(best, vocab)

            print(f"Merge {i+1}: {best}")

    # -----------------------------
    # Save merges
    # -----------------------------
    def save_merges(self, filename="merges.json"):
        with open(filename, "w") as f:
            json.dump(self.merges, f)

    # -----------------------------
    # Load merges
    # -----------------------------
    def load_merges(self, filename="merges.json"):
        with open(filename) as f:
            merges = json.load(f)
            self.merges = [tuple(x) for x in merges]

    # -----------------------------
    # Encode
    # -----------------------------
    def encode(self, text):

        words = text.split()
        encoded = []

        for word in words:

            symbols = list(word)
            symbols.append("</w>")

            for pair in self.merges:

                i = 0
                while i < len(symbols) - 1:

                    if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                        symbols[i:i + 2] = ["".join(pair)]
                    else:
                        i += 1

            encoded.append(symbols)

        return encoded

    # -----------------------------
    # Decode
    # -----------------------------
    def decode(self, tokens):

        words = []

        for token_list in tokens:
            word = ""

            for token in token_list:
                if token == "</w>":
                    continue
                word += token

            words.append(word)

        return " ".join(words)


# ----------------------------------------------------
# Example
# ----------------------------------------------------

training_corpus = [
    "low",
    "lowest",
    "newer",
    "wider",
    "low",
    "lower",
    "new",
    "widest",
    "slow",
]

bpe = BPETokenizer()

bpe.train(training_corpus, num_merges=15)

bpe.save_merges()

print("\nSaved merge rules to merges.json\n")

# ----------------------------------------------------
# Test words NOT in training
# ----------------------------------------------------

tests = [
    "lowering",
    "lowest",
    "low-cost"
]

for word in tests:

    encoded = bpe.encode(word)
    decoded = bpe.decode(encoded)

    print("=" * 40)
    print("Word   :", word)
    print("Encoded:", encoded)
    print("Decoded:", decoded)