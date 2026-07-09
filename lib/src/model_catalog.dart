// Copyright 2026 The Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import 'package:betto_onnxrt/betto_onnxrt.dart';

/// Allowlist of supported embedding models for dense retrieval.
///
/// [ModelCatalog] is the single place where models are registered and the
/// concrete [AllowlistProvider] implementation used with [ModelDownloader].
/// All models must appear here before they can be downloaded — attempting to
/// look up an unregistered model ID throws [ArgumentError]. Attempting to load
/// a model whose validation flag is `false` throws [UnsupportedError].
///
/// ## Why AllowlistProvider
///
/// [ModelCatalog] implements [AllowlistProvider] from `betto_onnxrt` so that
/// [ModelDownloader] can be constructed with `allowlist: ModelCatalog()` and
/// will reject any model not in this catalog before touching the network.
///
/// ## Adding a new model
///
/// 1. Add a private `ModelSpec` field below (as a `static final`).
/// 2. Insert it into the [_catalog] map with its ID as the key.
/// 3. Add `'<id>': false` to [_validated] until the model has been tested in
///    CI; flip it to `true` when the validation plan is complete.
///
/// ## Usage
///
/// ```dart
/// final spec = ModelCatalog.lookup('bge-small-en-v1.5');
/// print(spec.meta['dimensions']); // 384
///
/// // Use with ModelDownloader to gate downloads:
/// final downloader = ModelDownloader(allowlist: ModelCatalog());
/// ```
final class ModelCatalog implements AllowlistProvider {
  /// Creates a [ModelCatalog].
  ///
  /// The catalog is stateless and lightweight — create a new instance wherever
  /// needed, or share a single instance.
  const ModelCatalog();

  // ── Registered models ──────────────────────────────────────────────────────
  // Note: ModelSpec / ModelFile cannot be const because Uri(...) is not const
  // in Dart. Use static final (lazily initialised) instead.

  /// BGE Small En v1.5 (BAAI).
  ///
  /// 384-dimensional English-language sentence embeddings optimised for
  /// retrieval. ~127 MB ONNX binary. **Validated and production-ready.**
  static final _bgeSmallEnV15 = ModelSpec(
    id: 'bge-small-en-v1.5',
    files: {
      'onnx': ModelFile(
        url: Uri.parse(
          'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx',
        ),
        // SHA-256 of the exact model file used in CI; update if upstream
        // changes.
        sha256:
            '828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35',
      ),
      'vocab': ModelFile(
        url: Uri.parse(
          'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/vocab.txt',
        ),
        sha256:
            '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
      ),
    },
    meta: {
      // Embedding vector dimension. Read by OnnxEmbeddingModel as
      // `spec.meta['dimensions'] as int`.
      'dimensions': 384,
      // Selects BertTokenizer in OnnxEmbeddingModel.load(). Set explicitly
      // (not left absent) so a future third tokenizer family can't be
      // silently misresolved by a ModelSpec that forgot this key.
      'tokenizerFamily': 'bert',
      // No queryPrefix/documentPrefix — BGE Small En v1.5 has no
      // passage/query prefix convention, so OnnxEmbeddingModel.embed()'s
      // prefix step is a no-op for this model regardless of EmbeddingKind.
    },
  );

  /// `multilingual-e5-small` (intfloat) — multilingual, 384-dimensional.
  ///
  /// Same embedding dimension as [_bgeSmallEnV15], so adopting it requires no
  /// SQ8/index-format change in a consuming database. Registered as this
  /// project's first cross-lingual embedding model (WI-4); XLM-RoBERTa-family
  /// SentencePiece/Unigram tokenisation via [XlmRobertaTokenizer].
  ///
  /// **Registers the plain fp32 `model.onnx` export (~470 MB) — not a
  /// quantized or GPU-oriented variant.** `intfloat/multilingual-e5-small`'s
  /// `onnx/` directory also publishes `model_qint8_avx512_vnni.onnx`
  /// (x86 AVX512-VNNI int8, a poor fit for this project's largely-ARM
  /// targets) and `model_O4.onnx` (an ORT graph-optimizer export with fp16
  /// mixed precision intended for GPU inference, not the CPU-only execution
  /// providers `betto_onnxrt` supports). The plain export matches
  /// [_bgeSmallEnV15]'s own registration precedent (also a plain fp32
  /// `model.onnx`) and avoids stacking an unvalidated second source of
  /// numerical drift underneath the storage layer's own SQ8 quantization.
  /// Do not "helpfully" swap in a smaller quantized/optimized variant without
  /// a dedicated accuracy-validation pass — see
  /// `docs/plans/plan_0_06_wi4_multilingual_embedding_model.md` (in the
  /// `kmdb` repo) for the full trade-off analysis.
  ///
  /// **Note the meaningfully larger download** compared to BGE Small En v1.5
  /// (~470 MB vs. ~127 MB, ~3.7×) — worth surfacing in any first-use download
  /// UX, especially on mobile.
  static final _multilingualE5Small = ModelSpec(
    id: 'multilingual-e5-small',
    files: {
      'onnx': ModelFile(
        url: Uri.parse(
          'https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx',
        ),
        // SHA-256 of the exact model file used in CI; update if upstream
        // changes.
        sha256: _multilingualE5SmallOnnxSha256,
      ),
      'vocab': ModelFile(
        url: Uri.parse(
          'https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/tokenizer.json',
        ),
        sha256: _multilingualE5SmallTokenizerJsonSha256,
      ),
    },
    meta: {
      'dimensions': 384,
      // Selects XlmRobertaTokenizer in OnnxEmbeddingModel.load().
      'tokenizerFamily': 'xlmr',
      // multilingual-e5-small requires a mandatory "query: " / "passage: "
      // prefix per its model card — without it, retrieval quality degrades
      // silently (no error, just worse rankings). OnnxEmbeddingModel.embed()
      // applies these based on the caller's EmbeddingKind.
      'queryPrefix': 'query: ',
      'documentPrefix': 'passage: ',
    },
  );

  /// SHA-256 of `multilingual-e5-small`'s `onnx/model.onnx`, as downloaded
  /// and verified by `tool/register_model.dart` — see that script's own doc
  /// comment for how to regenerate/re-verify this value.
  static const _multilingualE5SmallOnnxSha256 =
      'ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665';

  /// SHA-256 of `multilingual-e5-small`'s `tokenizer.json`, as downloaded and
  /// verified by `tool/register_model.dart`.
  static const _multilingualE5SmallTokenizerJsonSha256 =
      '0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39';

  /// Internal fixture — **not a real model.**
  ///
  /// Exists solely so tests can exercise the catalog's "registered but not
  /// validated" gating behaviour (throws [UnsupportedError] from [lookup])
  /// against a stable id that will never be flipped to validated. Its file
  /// URLs deliberately point at a non-resolvable host so any accidental
  /// real-world use (someone actually trying to download or load it) fails
  /// fast and obviously rather than silently. See `plan_0_06_wi4_multilingual_embedding_model.md`'s
  /// Q5 for why this replaced the previous `bge-m3-v1.0` stub entry (which
  /// had placeholder all-zero checksums and could never actually be
  /// downloaded or validated).
  ///
  /// **Must never be set to `true` in [_validated].**
  static final _placeholderModel = ModelSpec(
    id: 'placeholder-model',
    files: {
      'onnx': ModelFile(
        url: Uri.parse('https://example.invalid/placeholder/model.onnx'),
        sha256:
            '0000000000000000000000000000000000000000000000000000000000000000',
      ),
      'vocab': ModelFile(
        url: Uri.parse('https://example.invalid/placeholder/vocab.txt'),
        sha256:
            '0000000000000000000000000000000000000000000000000000000000000000',
      ),
    },
    meta: {'dimensions': 0},
  );

  // ── Internal catalog and validation state ─────────────────────────────────

  /// All registered models keyed by [ModelSpec.id].
  ///
  /// Uses a lazy getter rather than a const map because the values are
  /// `static final` (not const, due to `Uri` not being const in Dart).
  static Map<String, ModelSpec> get _catalog => {
    'bge-small-en-v1.5': _bgeSmallEnV15,
    'multilingual-e5-small': _multilingualE5Small,
    'placeholder-model': _placeholderModel,
  };

  /// Validation state for each registered model.
  ///
  /// `betto_onnxrt`'s generic [ModelSpec] has no validation concept, so this
  /// catalog tracks it separately here. Only models explicitly set to `true`
  /// are permitted by [lookup]. A model absent from this map is considered
  /// unvalidated.
  static const Map<String, bool> _validated = {
    'bge-small-en-v1.5': true,
    'multilingual-e5-small': true,
    // Deliberately, permanently false — see _placeholderModel's doc comment.
    'placeholder-model': false,
  };

  // ── Public API ─────────────────────────────────────────────────────────────

  /// The ID of the default/recommended production model.
  static const String defaultModelId = 'bge-small-en-v1.5';

  /// Returns all registered [ModelSpec]s (validated and unvalidated).
  ///
  /// Useful for listing available models in a CLI command. Check
  /// the model's validation state via [_validated] before presenting it as
  /// user-selectable.
  static Iterable<ModelSpec> get all => _catalog.values;

  /// Looks up the [ModelSpec] for [id].
  ///
  /// Throws [ArgumentError] if [id] is not registered in the catalog.
  /// Throws [UnsupportedError] if the model is registered but not yet
  /// validated for production use.
  ///
  /// ```dart
  /// final spec = ModelCatalog.lookup('bge-small-en-v1.5');
  /// print(spec.meta['dimensions']); // 384
  /// ```
  static ModelSpec lookup(String id) {
    final catalog = _catalog;
    final spec = catalog[id];
    if (spec == null) {
      final known = catalog.keys.join(', ');
      throw ArgumentError(
        "Unknown embedding model ID '$id'. "
        "Registered models: $known. "
        "Add the model to ModelCatalog to use it.",
      );
    }
    if (!(_validated[id] ?? false)) {
      throw UnsupportedError(
        "Embedding model '$id' is registered in the catalog but has not "
        "yet been validated for production use. It will be enabled in "
        "a future release.",
      );
    }
    return spec;
  }

  /// Returns `true` if [id] is a known registered model ID (validated or not).
  ///
  /// Does **not** check validation status. Useful for detecting legacy config
  /// files that reference a known (but maybe unvalidated) model.
  static bool isKnown(String id) => _catalog.containsKey(id);

  // ── AllowlistProvider ──────────────────────────────────────────────────────

  /// Returns `true` if [spec] is registered in this catalog.
  ///
  /// Implements [AllowlistProvider] for use with [ModelDownloader]:
  ///
  /// ```dart
  /// final downloader = ModelDownloader(allowlist: ModelCatalog());
  /// ```
  ///
  /// This permits downloading of unvalidated models (e.g.
  /// `multilingual-e5-small` prior to its validation pass completing during
  /// development). Call [lookup] (which checks validation status) before
  /// loading a model for inference.
  @override
  bool isAllowed(ModelSpec spec) => _catalog.containsKey(spec.id);
}
