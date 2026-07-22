from bpe import get_pair_counts, merge_pair, split_words

words = split_words("low lower lowest")

for step in range(10):

    pairs = get_pair_counts(words)

    if not pairs:
        break

    best = max(pairs, key=pairs.get)

    print("Merge:", best)

    words = merge_pair(words, best)