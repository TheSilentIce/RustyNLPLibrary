# RustyNLPLibrary

A small NLP library in Rust built to learn how language models work end to end: a BPE tokenizer, Word2Vec embeddings, a decoder-only Transformer, and BM25 retrieval for RAG.

The model components (tokenization, embeddings, attention, layer norm, training loop, retrieval) are written by hand. [burn](https://github.com/tracel-ai/burn) is used only as the tensor and autodiff backend, running on WGPU.

## What's inside

| Component | File | What it does |
|---|---|---|
| **BPE tokenizer** | `components/tokenizer.rs` | Learns merges from a corpus, then encodes text into subword tokens |
| **Word2Vec** | `components/w2v.rs` | Skip-gram with negative sampling and batched training. Includes cosine similarity and analogy lookup |
| **Transformer** | `components/attention.rs` | Decoder-only, multi-head causal self-attention, pre/post LayerNorm, GELU feed-forward, sinusoidal positional embeddings. Trained with cross-entropy and SGD, with padding handled in the loss |
| **BM25 retrieval (RAG)** | `components/rag.rs` | Inverted index over a folder of `.txt` docs using the BPE tokenizer. Ranks documents with BM25 (tunable `k1`, `b`) and returns raw text to feed back into the model |
| **Neural net / logistic regression** | `components/neural_net.rs`, `logistic_regression.rs` | Building blocks used while developing the larger model |
| **N-grams** | `components/n_grams.rs` | Unigram and bigram counts, used for negative sampling |

**Work in progress:** GloVe embeddings (`glove.rs`) are stubbed out, and the RAG index isn't wired into the interactive generation loop yet.

## Pipeline

`main.rs` runs the full pipeline:

1. Learn a BPE vocabulary (2,000 merges) and tokenize the corpus
2. Pretrain Word2Vec embeddings
3. Initialize the Transformer with those embeddings (2 blocks, 4 heads, d = 64, context window 64)
4. Train
5. Open an interactive prompt that generates 20 tokens per input

## Running it

Requires a recent Rust toolchain (edition 2024) and a GPU supported by WGPU.

```bash
git clone https://github.com/TheSilentIce/RustyNLPLibrary.git
cd RustyNLPLibrary/rusty_lib
cargo run --release
```

Training text is read from the top-level `texts/` folder. `main.rs` expects a file named `TinyStories.txt` there. To use the included sample, rename `tinystories_sample.txt` or change the filename in `main.rs`.

## Using BM25 retrieval

```rust
use std::path::Path;
use rusty_lib::components::{rag::BM25Index, tokenizer::bpe_encoder};

let index = BM25Index::build_from_directory(Path::new("docs/"), &vocab, 1.2, 0.75)?;
let query = bpe_encoder(&vocab, &"how does attention work".to_string());

for (path, score) in index.search(&query, 3) {
    println!("{score:.3}  {path}");
    let context = index.get_text(&path); // raw text to prepend to the prompt
}
```

## Benchmarks

Criterion benchmarks cover tokenization, Word2Vec training and analogy accuracy, the neural net, a Transformer forward pass, and perplexity evaluation:

```bash
cargo bench --bench attention_bench
cargo bench --bench word2vec_bench
cargo bench --bench perplexity_bench
```
