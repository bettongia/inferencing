# Changelog

## 0.1.0-dev.2

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
