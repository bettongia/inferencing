# betto_inferencing

ONNX Runtime inference and embedding models for dense text retrieval. Part of
the [Bettongia](https://github.com/bettongia) open-source family.

Provides `OnnxEmbeddingModel` backed by either
[BGE Small En v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
(English-only) or
[`multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(cross-lingual) via [`betto_onnxrt`](https://github.com/bettongia/onnxrt), a
`ModelTokenizer` abstraction over BERT WordPiece and XLM-RoBERTa-family
SentencePiece/Unigram tokenization, SQ8 vector quantization helpers, and a
validated model catalog with download-on-demand.

## Features

- **`OnnxEmbeddingModel`** — loads an embedding model and embeds text into
  L2-normalised float32 vectors. Supports download-on-demand (SHA-256 verified)
  or loading from an explicit file path. Accepts an optional `tokenizer`
  parameter to swap in a custom word segmentation backend for BERT-family
  models (e.g. `IcuTokenizer` from `package:betto_icu`). Selects the concrete
  `ModelTokenizer` implementation via `ModelSpec.meta['tokenizerFamily']`.
- **`EmbeddingModel`** — abstract interface, allowing the consuming application
  or database to depend on the interface without the FFI-heavy implementation.
  `embed(text, {kind})` takes an `EmbeddingKind` (`document` or `query`, default
  `document`) so models with a mandatory passage/query prefix convention (e.g.
  `multilingual-e5-small`) apply the right one automatically.
- **`EmbeddingKind`** — `document` / `query`. Selects `ModelSpec.meta`'s
  `'documentPrefix'` / `'queryPrefix'`, if the loaded model defines them. A
  no-op for models with neither key (e.g. `bge-small-en-v1.5`).
- **`ModelTokenizer`** — shared interface implemented by `BertTokenizer` and
  `XlmRobertaTokenizer`, letting `OnnxEmbeddingModel` select either at runtime
  from a single `encode(String) -> TokenizerOutput` call site.
- **`BertTokenizer`** — BERT WordPiece tokenizer loaded from a `vocab.txt` file.
  Normalises, segments, and assembles `[CLS]`/`[SEP]`-padded token ID sequences
  ready for ORT inference. Word segmentation delegates to a `betto_lexical`
  `Tokenizer`; defaults to `RegExpTokenizer` and accepts `IcuTokenizer` as a
  drop-in replacement.
- **`TokenizerOutput`** — value type returned by `BertTokenizer.encode` and
  `XlmRobertaTokenizer.encode`. Carries `inputIds`, `attentionMask`, and
  `tokenTypeIds` as parallel `Int64List` arrays, plus a `truncated` flag.
- **`XlmRobertaTokenizer`** — XLM-RoBERTa-family SentencePiece/Unigram tokenizer
  (e.g. `multilingual-e5-small`), loaded from a HuggingFace `tokenizer.json`
  file. See "Why not `dart_sentencepiece_tokenizer` alone" below for why this
  isn't a thin wrapper around that package.
- **`ModelCatalog`** — allowlist of validated embedding models with
  download-on-demand gating. `ModelCatalog.lookup(id)` returns the `ModelSpec`
  for a validated model; `ModelCatalog.isKnown(id)` checks registration without
  validating; `ModelCatalog.defaultModelId` is `'bge-small-en-v1.5'`. Currently
  validated: `bge-small-en-v1.5` (384-d, English) and `multilingual-e5-small`
  (384-d, ~100 languages). `placeholder-model` is a permanent, never-validated
  internal test fixture (not a real model — see its doc comment in
  `lib/src/model_catalog.dart`).
- **`quantise` / `dequantise`** — SQ8 (scalar 8-bit) vector quantization helpers
  for compact storage of embedding indexes.
- Re-exports from `betto_onnxrt`: `ModelDownloader`, `ModelSpec`, `ModelFile`,
  `ResolvedModel`, `DownloadProgress`.

## Platform support

This package is **native-only**. It must not be imported on the web platform.

| Platform | Support                             | Notes                                                       |
| -------- | ----------------------------------- | ----------------------------------------------------------- |
| macOS    | arm64 only                          | Intel (x86_64) is not supported                             |
| Linux    | x64, aarch64                        |                                                             |
| Windows  | x64, arm64                          |                                                             |
| Android  | arm64-v8a, armeabi-v7a, x86_64, x86 | Requires `minSdkVersion 35`                                 |
| iOS      | arm64                               | Requires the `betto_onnxrt_ios` companion plugin (iOS ≥ 16) |
| Web      | Not supported                       |                                                             |

The ONNX Runtime shared library is staged at build time by the `betto_onnxrt`
native-assets build hook — no manual download or bundling is required.

### iOS setup

iOS ORT support requires the `betto_onnxrt_ios` Flutter plugin, which statically
links the ORT XCFramework into the host app via SPM. No CocoaPods changes are
needed. Add it alongside `betto_inferencing`:

```yaml
dependencies:
  betto_inferencing: ^0.1.0-dev.3
  betto_onnxrt_ios: ^0.1.0-dev.1
```

Requires Flutter ≥ 3.27.0 and iOS ≥ 16.

### Android

Set `minSdkVersion` to at least **35** in `android/app/build.gradle`:

```kotlin
android {
    defaultConfig {
        minSdk = 35
    }
}
```

## Installation

```yaml
dependencies:
  betto_inferencing: ^0.1.0-dev.3
```

> **Note:** `betto_inferencing` uses native-assets build hooks via
> `betto_onnxrt`. Run `dart test` (or `flutter test`) from **inside** the
> package directory (not the workspace root) so the hook fires and the ORT
> binary is placed correctly.

## Usage

### Embed text (download-on-demand)

```dart
import 'package:betto_inferencing/betto_inferencing.dart';

final model = await OnnxEmbeddingModel.load(
  cacheDir: '/path/to/model/cache',
  onProgress: (received, total) {
    final pct = total > 0 ? (received * 100 ~/ total) : 0;
    print('Downloading: $pct% ($received / $total bytes)');
  },
);

try {
  final (embedding, truncated) = await model.embed('semantic search query');
  print('Dimensions: ${embedding.length}'); // 384
  print('Truncated: $truncated');
} finally {
  model.dispose();
}
```

On the first call `ModelDownloader` fetches and SHA-256-verifies the BGE Small
En v1.5 model (~127 MB). Subsequent calls reuse the cached files.

> **Note:** `OnnxEmbeddingModel.load` throws `ArgumentError` if neither
> `modelPath` nor `cacheDir` is supplied — there is no bundled model asset.

### Embed text (explicit path)

```dart
final model = await OnnxEmbeddingModel.load(
  modelPath: '/path/to/model.onnx',
);
```

### Implement the interface

`EmbeddingModel` decouples the consuming application from this package's FFI
dependency:

```dart
import 'package:betto_inferencing/betto_inferencing.dart';

class MyRetriever {
  MyRetriever(this._model);

  final EmbeddingModel _model;

  Future<List<double>> queryVector(String text) async {
    final (embedding, _) = await _model.embed(text);
    return embedding.toList();
  }
}
```

### Look up a catalog model

```dart
// Lookup throws if the model is unknown or not yet validated.
final spec = ModelCatalog.lookup('bge-small-en-v1.5');
print(spec.meta['dimensions']); // 384

// Check registration without validation gating.
print(ModelCatalog.isKnown('placeholder-model')); // true (never validated)

// Iterate all registered models (validated and not yet validated).
for (final spec in ModelCatalog.all) {
  print('${spec.id} — ${spec.meta['dimensions']}d');
}
// bge-small-en-v1.5 — 384d
// multilingual-e5-small — 384d
// placeholder-model — 0d
```

### Cross-lingual embedding with `multilingual-e5-small`

```dart
final spec = ModelCatalog.lookup('multilingual-e5-small');
final model = await OnnxEmbeddingModel.load(
  spec: spec,
  cacheDir: '/path/to/model/cache',
);

try {
  // Index-time text: EmbeddingKind.document (the default) applies E5's
  // mandatory "passage: " prefix automatically.
  final (docEmbedding, _) = await model.embed(
    'The cat sat on the mat.',
    kind: EmbeddingKind.document,
  );

  // Query-time text in a different language: EmbeddingKind.query applies
  // "query: " instead.
  final (queryEmbedding, _) = await model.embed(
    'Le chat était assis sur le tapis.',
    kind: EmbeddingKind.query,
  );
} finally {
  model.dispose();
}
```

### SQ8 quantization

```dart
import 'package:betto_inferencing/betto_inferencing.dart';

final (embedding, _) = await model.embed('hello world');
final quantised = quantise(embedding);   // Uint8List, 384 bytes
final restored = dequantise(quantised);  // Float32List
```

## Examples

The `example/` directory contains runnable examples that walk through each
feature in the package. Set `BETTO_CACHE` to a persistent directory so the model
(~127 MB) is downloaded only once across all examples:

```sh
export BETTO_CACHE=$HOME/.cache/betto_examples
```

| File                                                       | What it shows                                                                                                             | Download required |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| [`example.dart`](example/example.dart)                     | Basic load + single `embed` call with download-on-demand                                                                  | Yes               |
| [`model_catalog.dart`](example/model_catalog.dart)         | List all registered models, inspect `ModelSpec` metadata, `isKnown`, and error handling for unknown / unvalidated IDs     | No                |
| [`tokenizer.dart`](example/tokenizer.dart)                 | `BertTokenizer` standalone: `encode`, token ID inspection, `decode`, WordPiece sub-token splitting, truncation detection  | Yes (vocab only)  |
| [`embed_and_compare.dart`](example/embed_and_compare.dart) | Embed a query and several documents, rank by cosine similarity (dot product of L2-normalised vectors)                     | Yes               |
| [`sq8_quantisation.dart`](example/sq8_quantisation.dart)   | `quantise`/`dequantise` round-trip: 4× storage reduction, per-element reconstruction error, similarity score preservation | Yes               |

Run any example with:

```sh
dart run example/<name>.dart
```

## Models

| Model ID                | Dimensions | Language        | Status                             |
| ------------------------ | ---------- | --------------- | ----------------------------------- |
| `bge-small-en-v1.5`      | 384        | English         | ✅ Validated (~127 MB download)     |
| `multilingual-e5-small`  | 384        | ~100 languages  | ✅ Validated (~470 MB download)     |
| `placeholder-model`      | —          | —               | Internal test fixture, never valid  |

`multilingual-e5-small` requires a `"passage: "` / `"query: "` text prefix
depending on whether the text is being indexed or queried — pass the matching
`EmbeddingKind` to `EmbeddingModel.embed`. `bge-small-en-v1.5` has no prefix
convention, so `EmbeddingKind` is a no-op for it (safe to omit).

Registering `bge-m3` (BAAI's larger, 1024-dimensional multilingual model) is
tracked as future work, not yet in this catalog — its ONNX export exceeds the
2 GB single-file limit and needs `ModelSpec`/`ModelDownloader` support for a
split `model.onnx` + `model.onnx_data` layout first.

## Why not `dart_sentencepiece_tokenizer` alone

`XlmRobertaTokenizer` depends on
[`dart_sentencepiece_tokenizer`](https://pub.dev/packages/dart_sentencepiece_tokenizer)
(MIT licensed — compatible with this project's Apache-2.0 license; no `NOTICE`
entry is needed for using it as a normal dependency) for vocabulary loading,
Unigram Viterbi decoding, and BOS/EOS post-processing, but cannot use it as-is
for the normalization step. Two independent defects were found in its
HuggingFace `tokenizer.json`-loading path while integrating
`multilingual-e5-small` (full investigation:
[`plan_0_06_wi11_xlmr_tokenizer.md`](https://github.com/bettongia/kmdb/blob/main/docs/plans/completed/plan_0_06_wi11_xlmr_tokenizer.md)
in the `kmdb` repository):

1. **It never applies the `Precompiled` charsmap normalizer on the JSON loading
   path.** `tokenizer.json`'s `normalizer` field is a `Sequence` whose first
   entry has `"type": "Precompiled"` and a `precompiled_charsmap` key holding a
   base64-encoded Darts double-array trie (the substitution table SentencePiece
   bakes into a trained model — fullwidth-to-ASCII folding, ellipsis expansion,
   etc.; not plain NFKC). `dart_sentencepiece_tokenizer`'s
   `huggingface_json.dart` parses this section's other flags correctly but never
   passes a `precompiledCharsmap` value through — it is silently dropped. (The
   charsmap data itself _is_ populated elsewhere, but only on the separate
   native `.model` protobuf loading path, which this project does not use — see
   the plan's Investigation section for the exact source lines.) No existing
   Dart implementation of this trie format could be found anywhere, so this
   package implements its own: `CharsmapTrie` (`lib/src/charsmap_trie.dart`),
   whose traversal algorithm structure is ported from the Rust `spm_precompiled`
   crate — see [`NOTICE`](NOTICE) for attribution.
2. **Its HuggingFace-JSON metadata parser also mis-derives
   whitespace/dummy-prefix configuration for `tokenizer.json` files that put it
   in `pre_tokenizer` rather than `normalizer`.** `multilingual-e5-small`'s
   `tokenizer.json` declares its dummy-prefix and whitespace-escaping behaviour
   entirely via a `pre_tokenizer.Metaspace` entry
   (`{"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}`),
   which the parser never reads (it only inspects `normalizer`) — so all three
   of its derived flags come back `false`, making its own `SpNormalizer` an
   unconditional pass-through for this file. This is a second, independent
   defect from the charsmap-drop above, found during this package's own
   investigation rather than documented upstream.

`XlmRobertaTokenizer` works around both by composing its own
charsmap-substitution, whitespace-collapse, and Metaspace-escaping steps
_before_ handing already-normalized text to `dart_sentencepiece_tokenizer`'s
public `encode()` — composition, not a fork: no `dart_sentencepiece_tokenizer`
source is modified. If either upstream defect is fixed in a future release, this
package's manual steps become redundant but harmless (idempotent no-ops on
already-correct input) — no urgency to remove them, though doing so is a
reasonable future cleanup.

## License

Apache 2.0 — see [LICENSE](LICENSE). Third-party attribution for ported
algorithm structure is recorded in [NOTICE](NOTICE).
