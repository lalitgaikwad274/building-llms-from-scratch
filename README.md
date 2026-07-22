# Build GPT From Scratch -- Phase 1 to Phase 3

This repository teaches how a GPT-style Large Language Model works by
implementing each component from scratch.

## Prerequisites

-   Python 3.10+
-   pip
-   Virtual environment (recommended)

## Project Structure

``` text
llm-from-scratch/
│
├── tokenizer/
│   ├── dataset.txt
│   ├── tokenizer.py
│   ├── train.py
│   ├── test.py
│   ├── bpe.py
│   ├── train_bpe.py
│   └── test_bpe.py
│
├── embeddings/
│   ├── embedding.py
│   └── main.py
│
├── attention/
│   ├── attention.py
│   ├── multi_head.py
│   └── main.py
│
└── README.md
```

## Setup

``` bash
python -m venv .venv
```

Linux/macOS:

``` bash
source .venv/bin/activate
```

Windows:

``` cmd
.venv\Scripts\activate
```

Install packages:

``` bash
pip install numpy regex tqdm matplotlib
```

------------------------------------------------------------------------

# Phase 1 -- Tokenizer

## Goal

Convert text into token IDs.

Pipeline:

``` text
Text
 ↓
Tokenizer
 ↓
Token IDs
```

### Files

-   `tokenizer.py` -- Tokenizer implementation
-   `train.py` -- Build vocabulary
-   `test.py` -- Encode/decode text

### Train

``` bash
cd tokenizer
python train.py
```

Expected:

-   `vocab.json` created

### Test

``` bash
python test.py
```

Expected output:

``` text
Encoded:
[3, 1, 0]

Decoded:
Hello AI .
```

------------------------------------------------------------------------

# Phase 1.2 -- Byte Pair Encoding (BPE)

## Goal

Learn sub-word tokens instead of whole words.

Pipeline:

``` text
Characters
    ↓
Merge frequent pairs
    ↓
Subword vocabulary
```

### Files

-   `bpe.py`
-   `train_bpe.py`
-   `test_bpe.py`

Run:

``` bash
python train_bpe.py
python test_bpe.py
```

------------------------------------------------------------------------

# Phase 2 -- Embeddings

## Goal

Convert token IDs into dense vectors.

Pipeline:

``` text
Token IDs
    ↓
Embedding Layer
    ↓
Vectors
```

Run:

``` bash
cd ../embeddings
python main.py
```

Expected shape:

``` text
(num_tokens, embedding_dimension)
```

------------------------------------------------------------------------

# Phase 3 -- Self Attention

## Goal

Allow every token to look at every other token.

Pipeline:

``` text
Embeddings
      ↓
Q, K, V
      ↓
Attention Scores
      ↓
Softmax
      ↓
Weighted Sum
```

Run:

``` bash
cd ../attention
python main.py
```

Verify:

-   Q shape
-   K shape
-   V shape
-   Attention matrix shape
-   Output shape

------------------------------------------------------------------------

# Phase 3.2 -- Multi-Head Attention

Pipeline:

``` text
Input
 ↓
Split into Heads
 ↓
Attention (per head)
 ↓
Concatenate
 ↓
Output Projection
```

Verify:

-   `embed_dim % num_heads == 0`
-   Output shape matches input embedding dimension

------------------------------------------------------------------------

# Learning Outcomes

By completing these phases you will understand:

-   Tokenization
-   Byte Pair Encoding
-   Embeddings
-   Query / Key / Value
-   Scaled Dot Product Attention
-   Softmax
-   Multi-Head Attention

These are the core building blocks of GPT-style models.

------------------------------------------------------------------------

# What's Next

The next phase is the **Transformer Block**, where Multi-Head Attention
is combined with:

1.  Layer Normalization
2.  Residual Connections
3.  Feed Forward Network

This creates the repeating block used in GPT, Llama, and most modern
LLMs.