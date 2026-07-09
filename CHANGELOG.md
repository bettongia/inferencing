# Changelog

## 0.1.0-dev.3

### Features

- **`multilingual-e5-small`** registered in `ModelCatalog` and validated —
  this package's first cross-lingual embedding model, ~100 languages, 384
  dimensions (same as `bge-small-en-v1.5`, so no SQ8/index-format change for
  consumers switching models). Uses the plain fp32 `model.onnx` export (not
  a quantized or GPU-oriented variant — see the `ModelCatalog` doc comment
  for the trade-off analysis) and `XlmRobertaTokenizer` for tokenization.
  Real checksums verified via the new `tool/register_model.dart`.
- **`EmbeddingKind`** — new `document` / `query` enum, and a corresponding
  `kind` parameter on `EmbeddingModel.embed()` (default `EmbeddingKind.document`,
  source-compatible for existing callers). `OnnxEmbeddingModel.embed()` uses
  it to apply `ModelSpec.meta`'s `'documentPrefix'` / `'queryPrefix'` when the
  loaded model defines them — `multilingual-e5-small` requires a mandatory
  `"passage: "` / `"query: "` prefix per its model card; `bge-small-en-v1.5`
  has neither key, so its behaviour is byte-for-byte unchanged.
- **`ModelTokenizer`** — new shared interface (`encode(String) ->
  TokenizerOutput`) implemented by both `BertTokenizer` and
  `XlmRobertaTokenizer`. `OnnxEmbeddingModel.load()` selects the concrete
  implementation via a new `ModelSpec.meta['tokenizerFamily']` key (`'bert'`
  or `'xlmr'`, added explicitly to `bge-small-en-v1.5`'s own entry too) rather
  than a compile-time fork, so a third tokenizer family is additive.
- **`tool/register_model.dart`** — new one-off dev tool that downloads a
  `ModelCatalog` entry's asset files from the exact URL `ModelSpec` uses at
  runtime and computes their SHA-256 digests via `package:crypto`, so a
  registered checksum is guaranteed to match what a real download verifies.

### Changes

- **`placeholder-model`** replaces the previous `bge-m3-v1.0` stub entry in
  `ModelCatalog`. The old entry had placeholder all-zero checksums and could
  never actually be downloaded or validated — a registered model that would
  silently fail with a confusing checksum-mismatch error rather than a clear
  "not supported" message. `placeholder-model` is a permanent, deliberately
  `validated: false` test fixture (non-resolvable `example.invalid` URLs)
  that exists solely to give tests a stable, always-unvalidated registered id
  to assert catalog gating behaviour against. Registering `bge-m3` properly
  (a real 1024-dimensional multilingual model) is deferred future work — its
  ONNX export exceeds the 2 GB single-file limit and needs
  `ModelSpec`/`ModelDownloader` support for a split `model.onnx` +
  `model.onnx_data` layout first.

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
