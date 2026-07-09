# Changelog

## 0.1.0-dev.2

### Features

- **`XlmRobertaTokenizer`** — XLM-RoBERTa-family SentencePiece/Unigram
  tokenizer (e.g. `multilingual-e5-small`), returning the same
  `TokenizerOutput` type `BertTokenizer` uses. Composes a from-scratch
  `CharsmapTrie` (a Darts double-array trie reader for SentencePiece's
  `precompiled_charsmap` normalizer) with `dart_sentencepiece_tokenizer`'s
  public API, working around two defects in that package's HuggingFace
  `tokenizer.json` loading path — see `README.md` for details and `NOTICE`
  for third-party attribution.
- **`tool/generate_xlmr_parity_corpus.dart`** and
  `test/fixtures/xlmr_parity_corpus.json` — a 58-language (+3 edge case)
  byte-exact tokenizer parity corpus, extracted from the NLTK UDHR corpus
  (same source `betto_lang_detector` uses) and annotated with real
  `AutoTokenizer` token ids. Gated by a new test in the `test-macos`
  integration suite, additive to the smaller 11-entry gate already in
  place.

### Fixes

- **`XlmRobertaTokenizer`** — empty-string input no longer produces a
  spurious extra token. `_metaspace` was unconditionally adding a
  dummy-prefix space before replacing spaces with `▁`, so `""` became
  `"▁"` — itself a valid standalone vocabulary piece — yielding `[<s>, ▁,
  </s>]` instead of the real `AutoTokenizer`'s `[<s>, </s>]`. Found by the
  new 61-entry parity corpus above.

## 0.1.0-dev.1

Initial development release providing ONNX Runtime inference and embedding
models for dense text retrieval on native platforms (macOS, Linux, Windows,
Android, iOS).

### Features

- **`EmbeddingModel`** — abstract interface for text-to-vector embedding,
  decoupling consumers from any specific inference backend.
- **`OnnxEmbeddingModel`** — ONNX Runtime implementation backed by BGE Small En
  v1.5, delivering dense embeddings suitable for semantic search and retrieval
  tasks.
- **`BertTokenizer` / `TokenizerOutput`** — BERT WordPiece tokenizer with
  configurable word segmentation (default `RegExpTokenizer`; drop-in
  `IcuTokenizer` support via `package:betto_icu`).
- **`ModelCatalog`** — allowlist provider that gates model use behind
  download-on-demand via `ModelDownloader` from `betto_onnxrt`. Registered
  models: `bge-small-en-v1.5` (validated), `bge-m3-v1.0` (registered, not yet
  validated — throws `UnsupportedError`).
- **`quantise` / `dequantise`** — SQ8 scalar quantization helpers for compact
  vector storage and retrieval.
- Re-exports `DownloadProgress`, `ModelDownloader`, `ModelFile`, `ModelSpec`,
  and `ResolvedModel` from `betto_onnxrt` as part of the stable public API.
