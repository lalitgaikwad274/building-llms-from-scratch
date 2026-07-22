from collections import Counter

def split_words(text):
    words = text.split()

    return [
        list(word)
        for word in words
    ]

def get_pair_counts(words):

    pairs = Counter()
     
    for word in words:
        for i in range(len(word)-1):

            pair = (
                word[i],
                word[i+1]
            )

            pairs[pair] += 1

    return pairs

def merge_pair(words, pair):

    new_words = []

    for word in words:

        merged = []

        i = 0

        while i < len(word):

            if (
                i < len(word)-1
                and word[i] == pair[0]
                and word[i+1] == pair[1]
            ):

                merged.append(
                    pair[0] + pair[1]
                )

                i += 2

            else:

                merged.append(word[i])

                i += 1

        new_words.append(merged)

    return new_words