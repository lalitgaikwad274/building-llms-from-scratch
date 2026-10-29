import os
import sys

# Ensure both local directory and project root are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
for path in [current_dir, parent_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from bpe import get_pair_counts, merge_pair, split_words
except ImportError:
    from tokenizer.bpe import get_pair_counts, merge_pair, split_words


def encode_word(word, merges):
    """Tokenize a single word into subwords using learned BPE merges."""
    tokens = split_words(word)
    for pair in merges:
        tokens = merge_pair(tokens, pair)
    return tokens[0] if tokens else []


def encode_text(text, merges):
    """Tokenize full text into a list of subwords for each word."""
    token_words = split_words(text)
    for pair in merges:
        token_words = merge_pair(token_words, pair)
    return token_words


def decode_tokens(token_words):
    """Reconstruct text from tokenized subwords."""
    return " ".join("".join(tokens) for tokens in token_words)


# ----------------------------------------------------
# Test 1: split_words
# ----------------------------------------------------
def test_split_words():
    print("=" * 60)
    print("1. Testing split_words()")
    print("=" * 60)

    text = "low lower lowest"
    words = split_words(text)
    expected = [
        ["l", "o", "w"],
        ["l", "o", "w", "e", "r"],
        ["l", "o", "w", "e", "s", "t"],
    ]

    print(f"Input text : '{text}'")
    print(f"Split words: {words}")
    assert words == expected, f"Expected {expected}, got {words}"
    print("✓ split_words test passed!\n")


# ----------------------------------------------------
# Test 2: get_pair_counts
# ----------------------------------------------------
def test_get_pair_counts():
    print("=" * 60)
    print("2. Testing get_pair_counts()")
    print("=" * 60)

    words = [["l", "o", "w"], ["l", "o", "w", "e", "r"]]
    pairs = get_pair_counts(words)

    print(f"Input words: {words}")
    print(f"Pair counts: {dict(pairs)}")

    assert pairs[("l", "o")] == 2, f"Expected ('l', 'o') count 2, got {pairs[('l', 'o')]}"
    assert pairs[("o", "w")] == 2, f"Expected ('o', 'w') count 2, got {pairs[('o', 'w')]}"
    assert pairs[("w", "e")] == 1, f"Expected ('w', 'e') count 1, got {pairs[('w', 'e')]}"
    assert pairs[("e", "r")] == 1, f"Expected ('e', 'r') count 1, got {pairs[('e', 'r')]}"
    print("✓ get_pair_counts test passed!\n")


# ----------------------------------------------------
# Test 3: merge_pair
# ----------------------------------------------------
def test_merge_pair():
    print("=" * 60)
    print("3. Testing merge_pair()")
    print("=" * 60)

    words = [["l", "o", "w"], ["l", "o", "w", "e", "r"]]

    # Merge ('l', 'o') -> 'lo'
    merged1 = merge_pair(words, ("l", "o"))
    print(f"Merge ('l', 'o')  : {merged1}")
    assert merged1 == [["lo", "w"], ["lo", "w", "e", "r"]]

    # Merge ('lo', 'w') -> 'low'
    merged2 = merge_pair(merged1, ("lo", "w"))
    print(f"Merge ('lo', 'w') : {merged2}")
    assert merged2 == [["low"], ["low", "e", "r"]]
    print("✓ merge_pair test passed!\n")


# ----------------------------------------------------
# Test 4: BPE Training and Subword Encoding/Decoding
# ----------------------------------------------------
def test_bpe_training_and_tokenization():
    print("=" * 60)
    print("4. Testing BPE Training & Subword Tokenization")
    print("=" * 60)

    training_text = "low lowest newer wider low lower new widest slow"
    print(f"Training corpus: '{training_text}'")

    words = split_words(training_text)
    merges = []
    num_merges = 10

    print("\nLearning BPE merges:")
    for step in range(num_merges):
        pairs = get_pair_counts(words)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        words = merge_pair(words, best)
        print(f"  Step {step + 1:2d} Merge: {best}")

    assert len(merges) > 0, "No merges learned!"

    # Test encoding on unseen words and subword splitting
    print("\nTesting Subword Encoding & Reconstruction:")
    test_words = [
        "lowering",
        "lowest",
        "low-cost",
        "newer",
        "widest",
        "slow",
    ]

    for word in test_words:
        subwords = encode_word(word, merges)
        reconstructed = "".join(subwords)
        print(f"  Word: {word:10s} -> Subwords: {subwords} -> Decoded: '{reconstructed}'")
        assert reconstructed == word, f"Reconstruction failed for '{word}': got '{reconstructed}'"

    # Test full sentence encode and decode
    sentence = "the lowest newer wider machine learning"
    encoded = encode_text(sentence, merges)
    decoded = decode_tokens(encoded)
    print(f"\nSentence : '{sentence}'")
    print(f"Encoded  : {encoded}")
    print(f"Decoded  : '{decoded}'")
    assert decoded == sentence, f"Sentence roundtrip failed: '{decoded}' != '{sentence}'"

    print("✓ BPE training & tokenization test passed!\n")


def run_all_tests():
    test_split_words()
    test_get_pair_counts()
    test_merge_pair()
    test_bpe_training_and_tokenization()

    print("=" * 60)
    print("All BPE tests passed successfully! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

