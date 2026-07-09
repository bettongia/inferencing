---
title: Technical Specification
subtitle: betto_inferencing
toc-title: "Contents"
...

- **Package:** `betto_inferencing`
- **Version:** 0.1.0-dev.3
- **Dart SDK:** ^3.12.0

# Purpose and scope

`betto_inferencing` provides ONNX Runtime-backed text embedding for dense
retrieval inside the Bettongia knowledge workspace. It is a **native-only**
pure-Dart package (no Flutter dependency) that:

- wraps the `betto_onnxrt` ONNX Runtime FFI binding into a high-level
  `EmbeddingModel` interface, with an `EmbeddingKind` parameter distinguishing
  index-time ("document") from query-time ("query") text;
- ships a `ModelTokenizer` abstraction implemented by a BERT WordPiece
  tokenizer (`BertTokenizer`, suitable for BGE-family models) and an
  XLM-RoBERTa-family SentencePiece/Unigram tokenizer (`XlmRobertaTokenizer`,
  suitable for `multilingual-e5-small`), selected at runtime by
  `OnnxEmbeddingModel.load` based on `ModelSpec.meta['tokenizerFamily']`;
- registers and gates supported models via `ModelCatalog`;
- provides SQ8 scalar quantization helpers (`quantise`/`dequantise`) for
  compact index storage.

The package does **not** implement vector search, BM25, or any retrieval
pipeline. Those concerns belong to the consuming layer.

# Architecture

## Layer position

```
┌─────────────────────────────────────┐
│  consuming layer (betto_db / app)   │
│  depends on EmbeddingModel only     │
└────────────────┬────────────────────┘
                 │ interface
┌────────────────▼────────────────────┐
│         betto_inferencing           │
│  OnnxEmbeddingModel                 │
│  BertTokenizer · XlmRobertaTokenizer│
│  ModelCatalog · SQ8 helpers         │
└────────────────┬────────────────────┘
                 │ FFI
┌────────────────▼────────────────────┐
│           betto_onnxrt              │
│  OnnxRuntime · OnnxSession          │
│  ModelDownloader · ModelSpec        │
└─────────────────────────────────────┘
```

The consuming layer is intended to depend only on `EmbeddingModel` so that it
carries no transitive FFI dependency. The concrete `OnnxEmbeddingModel` is
wired in at the application composition root.

## Embedding pipeline

```
text
  │
  ▼  BertTokenizer.encode()
TokenizerOutput (inputIds, attentionMask, tokenTypeIds)
  │
  ▼  OnnxSession.run()
last_hidden_state  [1, seqLen, D]
  │
  ▼  meanPool()  — average over non-padding token positions
pooled  [D]
  │
  ▼  l2Normalize()
embedding  [D]  unit-norm Float32List
```

# Dependencies

| Package | Role |
|---|---|
| `betto_onnxrt` | ORT FFI binding, `ModelDownloader`, `ModelSpec` |
| `betto_lexical` | `Tokenizer` interface, `RegExpTokenizer` |
| `dart_sentencepiece_tokenizer` | Vocabulary loading, Unigram Viterbi, BOS/EOS post-processing inside `XlmRobertaTokenizer` |
| `crypto` | SHA-256 verification in `ModelDownloader` |
| `path` | Path manipulation in `OnnxEmbeddingModel.load` |
| `meta` | `@visibleForTesting` annotations |
| `characters` | Grapheme-aware iteration inside `CharsmapTrie` |

`betto_lexical` supplies the word segmentation abstraction used inside
`BertTokenizer`. The default implementation is `RegExpTokenizer`; callers can
substitute `IcuTokenizer` from `package:betto_icu` for superior Unicode
coverage.

# Public API

## `EmbeddingModel`

Abstract interface in `lib/src/embedding_model.dart`. Exported from the
top-level library.

```
EmbeddingModel
  String get modelId
  int get dimensions
  Future<(Float32List embedding, bool truncated)> embed(
    String text, {
    EmbeddingKind kind = EmbeddingKind.document,
  })
  void dispose()
```

**Contract**

- `embed` must be safe to call from the main isolate.
- The returned `Float32List` has exactly `dimensions` elements.
- Empty or whitespace-only input must not throw. Behaviour is otherwise
  implementation-defined — `OnnxEmbeddingModel` produces a real
  `[CLS][SEP]`-only embedding (not a zero vector) with `truncated = false`. A
  model with a mandatory prefix (e.g. `multilingual-e5-small`) never actually
  tokenises truly empty content, since the prefix is prepended first.
- `kind` states whether `text` is being indexed (`EmbeddingKind.document`,
  the default) or is a search query (`EmbeddingKind.query`). Models with a
  passage/query prefix convention (e.g. `multilingual-e5-small`) apply the
  matching prefix based on `kind`; models without one (e.g.
  `bge-small-en-v1.5`) ignore it. Callers that know which case they're in
  should always pass the matching `kind` explicitly — silently defaulting
  degrades retrieval quality for models that use the distinction, without
  erroring.
- `dispose` must be called exactly once when the model is no longer needed.
  Calling `embed` after `dispose` is undefined behaviour.

## `EmbeddingKind`

Enum in `lib/src/embedding_model.dart`: `document` / `query`. Selects
`ModelSpec.meta`'s `'documentPrefix'` / `'queryPrefix'` inside
`OnnxEmbeddingModel.embed`, if the loaded model's spec defines them (see
`ModelCatalog` below). A no-op for models with neither key.

## `ModelTokenizer`

Shared interface in `lib/src/model_tokenizer.dart`, implemented by
`BertTokenizer` and `XlmRobertaTokenizer`:

```
ModelTokenizer
  TokenizerOutput encode(String text)
```

Both tokenizer families return the same concrete `TokenizerOutput` type, so
`OnnxEmbeddingModel` has a single `_tokenizer.encode(text)` call site
regardless of which family is loaded. `OnnxEmbeddingModel.load` selects the
concrete implementation from `ModelSpec.meta['tokenizerFamily']` (`'bert'` or
`'xlmr'`) — the key must be present and recognised; an absent or unrecognised
value throws `ArgumentError` synchronously, before any I/O.

## `OnnxEmbeddingModel`

Concrete implementation in `lib/src/onnx_embedding_model.dart`.

### Factory

```dart
static Future<OnnxEmbeddingModel> load({
  ModelSpec? spec,
  String? cacheDir,
  String? modelPath,
  Tokenizer? tokenizer,
  DownloadProgress? onProgress,
})
```

**Resolution rules**

| Supplied | Behaviour |
|---|---|
| `cacheDir` only | Download-on-demand using `ModelCatalog.defaultModelId` |
| `cacheDir` + `spec` | Download-on-demand using the given spec |
| `modelPath` only | Load from disk; identity from `ModelCatalog.defaultModelId` |
| `modelPath` + `spec` | Load from disk; identity from supplied spec |
| neither | Throws `ArgumentError` synchronously |

`tokenizer` (word-segmentation override) only applies when the resolved
spec's `tokenizerFamily` is `'bert'`; ignored for `'xlmr'` models, which have
no equivalent seam.

**Errors**

- `ArgumentError` — neither `modelPath` nor `cacheDir` supplied, or the
  resolved spec's `meta['tokenizerFamily']` is missing or not one of `'bert'`
  / `'xlmr'`. Both checks run synchronously, before any I/O.
- `UnsupportedError` — model file not found on disk.
- `Exception` — ORT library cannot be loaded or model is corrupt.

### Thread safety

ORT sessions are thread-affine. All `embed` and `dispose` calls must come from
the same isolate that called `load`. For Flutter UI threads, wrap calls in
`Isolate.run` — but create the session inside the spawned isolate.

### ONNX inputs / outputs

Both registered model families require three `int64` inputs shaped
`[1, seqLen]`:

| Input name | Source |
|---|---|
| `input_ids` | `TokenizerOutput.inputIds` |
| `attention_mask` | `TokenizerOutput.attentionMask` |
| `token_type_ids` | `TokenizerOutput.tokenTypeIds` |

The single output `last_hidden_state` has shape `[1, seqLen, D]`. The model
mean-pools over non-padding positions then L2-normalises to produce the final
embedding.

## `BertTokenizer`

BERT WordPiece tokenizer in `lib/src/bert_tokenizer.dart`. Implements
`ModelTokenizer`; selected by `OnnxEmbeddingModel.load` for models whose spec
sets `meta['tokenizerFamily'] = 'bert'` (e.g. `bge-small-en-v1.5`).

### Tokenization pipeline

1. **Normalize** — lower-case; strip Unicode combining diacritical marks
   (U+0300–U+036F).
2. **Word segmentation** — delegate to the `Tokenizer` supplied at
   construction (default: `RegExpTokenizer`).
3. **WordPiece** — greedily match longest sub-word pieces from `vocab.txt`.
   Sub-word continuations are prefixed with `##`. Unknown pieces map to
   `[UNK]` (ID 100).
4. **Assemble** — prepend `[CLS]` (101), append `[SEP]` (102), pad to
   `maxLength` with `[PAD]` (0).

### Special token IDs

| Constant | ID | Purpose |
|---|---|---|
| `clsId` | 101 | Start-of-sequence marker |
| `sepId` | 102 | End-of-segment marker |
| `unkId` | 100 | Unknown sub-word |
| `padId` | 0 | Padding |

### Truncation

The usable token budget is `maxLength - 2` (510 for the default
`maxLength = 512`). Tokens beyond this budget are silently discarded;
`TokenizerOutput.truncated` is set to `true`.

### `TokenizerOutput`

Value type returned by `BertTokenizer.encode`. All three arrays have exactly
`maxLength` elements.

| Field | Type | Description |
|---|---|---|
| `inputIds` | `Int64List` | BERT token IDs |
| `attentionMask` | `Int64List` | 1 for real tokens, 0 for padding |
| `tokenTypeIds` | `Int64List` | All zeros (single-segment input) |
| `truncated` | `bool` | True if input exceeded token budget |

## `XlmRobertaTokenizer`

XLM-RoBERTa-family SentencePiece/Unigram tokenizer in
`lib/src/xlmr_tokenizer.dart`, e.g. for `multilingual-e5-small`. Implements
`ModelTokenizer`; selected by `OnnxEmbeddingModel.load` for models whose spec
sets `meta['tokenizerFamily'] = 'xlmr'`. Returns the same `TokenizerOutput`
type `BertTokenizer` does (`tokenTypeIds` all-zero, since XLM-RoBERTa has no
segment ids; padding uses the loaded vocabulary's own pad id rather than
BERT's `padId = 0`).

Composes a from-scratch `CharsmapTrie` normalizer (`lib/src/charsmap_trie.dart`
— a Darts double-array trie reader for SentencePiece's `precompiled_charsmap`
normalizer, ported from the Rust `spm_precompiled` crate; see `NOTICE`) with
`dart_sentencepiece_tokenizer`'s public API for vocabulary loading, Unigram
Viterbi decoding, and `<s>`/`</s>` post-processing. See this package's
`README.md` ("Why not `dart_sentencepiece_tokenizer` alone") for the full
rationale and the two upstream defects this works around.

## `ModelCatalog`

Registered model allowlist in `lib/src/model_catalog.dart`. Implements
`AllowlistProvider` from `betto_onnxrt` so it can be passed directly to
`ModelDownloader(allowlist: ModelCatalog())`.

### Registered models

| Model ID | Dimensions | Language | tokenizerFamily | Status |
|---|---|---|---|---|
| `bge-small-en-v1.5` | 384 | English | `bert` | Validated (~127 MB download) |
| `multilingual-e5-small` | 384 | ~100 languages | `xlmr` | Validated (~470 MB download) |
| `placeholder-model` | — | — | — | Internal test fixture, never validated |

`multilingual-e5-small`'s spec also carries `meta['queryPrefix'] = 'query: '`
and `meta['documentPrefix'] = 'passage: '` (see `EmbeddingKind` above).
`bge-small-en-v1.5` has neither key.

`placeholder-model` is not a real model — it exists solely so tests can
exercise the "registered but not validated" gating path (`UnsupportedError`
from `lookup`) against a stable id that will never be flipped to validated.
Its file URLs point at a non-resolvable host (`example.invalid`) so a misuse
fails fast. It replaced a previous `bge-m3-v1.0` stub entry whose checksums
were unverifiable placeholders; registering BGE-M3 properly (a genuinely
larger, 1024-dimensional multilingual model) is tracked as separate future
work — its ONNX export exceeds the 2 GB single-file limit and needs
`ModelSpec`/`ModelDownloader` support for a split `model.onnx` +
`model.onnx_data` layout first.

### Public API

| Member | Description |
|---|---|
| `ModelCatalog.defaultModelId` | `'bge-small-en-v1.5'` |
| `ModelCatalog.all` | All registered `ModelSpec`s |
| `ModelCatalog.lookup(id)` | Returns spec; throws if unknown or not yet validated |
| `ModelCatalog.isKnown(id)` | True if registered (ignores validation status) |
| `isAllowed(spec)` | `AllowlistProvider` — true if registered (used by downloader) |

`lookup` throws `ArgumentError` for unknown IDs and `UnsupportedError` for
registered-but-not-yet-validated models. `isAllowed` intentionally permits
downloading not-yet-validated models during development; call `lookup` (which
checks validation) before running inference.

## SQ8 quantization

Functions in `lib/src/sq8.dart`. Exported from the top-level library.

### `quantise(Float32List) → Uint8List`

Maps each L2-normalised float component from `[-1.0, 1.0]` to `[0, 255]`:

```
u = clamp(round((f + 1.0) / 2.0 * 255), 0, 255)
```

Maximum quantization error per component: `2.0 / 255 ≈ 0.00784`.

### `dequantise(Uint8List) → Float32List`

Inverse mapping:

```
f = u / 255.0 * 2.0 - 1.0
```

The reconstructed vector is no longer unit-norm. For cosine similarity via
dot product this is acceptable — ranking order is preserved.

**Assumption:** Input to `quantise` must be L2-normalised (all components in
`[-1.0, 1.0]`). Values marginally outside the range due to float rounding are
clamped.

## Re-exported types from `betto_onnxrt`

The following types are re-exported from `betto_inferencing` as part of its
stable public surface:

| Type | Purpose |
|---|---|
| `ModelSpec` | Descriptor for a downloadable model |
| `ModelFile` | URL and SHA-256 for a single model file |
| `ModelDownloader` | Downloads and SHA-256-verifies model files |
| `ResolvedModel` | Result of a successful download/cache hit |
| `DownloadProgress` | Callback type `void Function(int received, int total)` |

# Platform support

| Platform | Supported | Notes |
|---|---|---|
| macOS | arm64 only | Intel (x86_64) throws `UnsupportedError` at build time |
| Linux | x64, aarch64 | ORT shared object via native-assets hook |
| Windows | x64, arm64 | ORT DLL + companion DLL via native-assets hook |
| Android | arm64-v8a, x86_64, armeabi-v7a, x86 | Requires `minSdkVersion 35` |
| iOS | arm64 | Requires the `betto_onnxrt_ios` Flutter plugin (iOS ≥ 16) |
| Web | Not supported | No FFI; must not be imported |

The ORT binary is staged at build time by the `betto_onnxrt` native-assets
build hook (`hook/build.dart` in that package). No manual download or bundling
is required on any supported platform.

## iOS detail

iOS uses ORT version **1.24.2** (via `onnxruntime-swift-package-manager`),
which is higher than the desktop/Android version (1.22.0). The ORT C API is
append-only: requesting API version 22 from ORT 1.24.2 returns the same vtable
as 1.22.x, so the runtime behaviour is identical.

The `betto_onnxrt_ios` Flutter plugin declares an SPM dependency on
`microsoft/onnxruntime-swift-package-manager` (product `onnxruntime`). Xcode
statically links the ORT XCFramework into the host app binary. At runtime
`OnnxRuntime.load()` calls `DynamicLibrary.process()` to resolve ORT C API
symbols from the process image. No CocoaPods or Podfile changes are needed.
`OnnxRuntime.dispose()` skips `DynamicLibrary.close()` on iOS because
`DynamicLibrary.process()` represents the process image and cannot be closed.

Add `betto_onnxrt_ios` alongside `betto_inferencing` in the consuming Flutter
app's `pubspec.yaml`:

```yaml
dependencies:
  betto_inferencing: ^0.1.0-dev.1
  betto_onnxrt_ios: ^0.1.0-dev.2
```

Requires Flutter ≥ 3.27.0.

## Android detail

The build hook downloads the ORT Android AAR from Maven Central and extracts
the per-ABI `.so`. Two-level SHA-256 verification is applied: the AAR archive
itself, then the extracted `.so`. Set `minSdkVersion` to at least **35** in
`android/app/build.gradle` (or `build.gradle.kts`):

```kotlin
android {
    defaultConfig {
        minSdk = 35
    }
}
```

# Error handling

| Condition | Type | Where thrown |
|---|---|---|
| Neither `modelPath` nor `cacheDir` supplied | `ArgumentError` | `OnnxEmbeddingModel.load` |
| `tokenizerFamily` missing or unrecognised | `ArgumentError` | `OnnxEmbeddingModel.load` |
| Model file not found on disk | `UnsupportedError` | `OnnxEmbeddingModel.load` |
| Unknown model ID in catalog | `ArgumentError` | `ModelCatalog.lookup` |
| Registered but not yet validated model | `UnsupportedError` | `ModelCatalog.lookup` |
| ORT library load failure | `Exception` | `OnnxRuntime.load` (betto_onnxrt) |
| ONNX model corrupt or mismatched | `StateError` | `OnnxTensor.asFloat32` |

# Internal utilities (not exported)

`lib/src/math_utils.dart` provides internal math helpers used by
`OnnxEmbeddingModel.embed`. They are not part of the public API.

| Function | Description |
|---|---|
| `meanPool(hiddenState, attentionMask, {seqLen, hiddenDim})` | Average token hidden states weighted by attention mask |
| `l2Normalize(vec)` | Normalise a float32 vector to unit L2 norm in-place |
| `cosineSimilarity(a, b)` | Dot product of two unit-norm vectors |
