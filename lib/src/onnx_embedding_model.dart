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

import 'dart:io';
import 'dart:typed_data';

import 'package:betto_onnxrt/betto_onnxrt.dart';
import 'package:betto_lexical/betto_lexical.dart' show Tokenizer;
import 'package:meta/meta.dart' show visibleForTesting;
import 'package:path/path.dart' as p;

import 'embedding_model.dart' show EmbeddingKind, EmbeddingModel;

import 'bert_tokenizer.dart';
import 'math_utils.dart';
import 'model_catalog.dart';
import 'model_tokenizer.dart' show ModelTokenizer;
import 'xlmr_tokenizer.dart' show XlmRobertaTokenizer;

/// ONNX Runtime-backed embedding model for dense text retrieval.
///
/// Implements [EmbeddingModel] using a model from [ModelCatalog] via the
/// `betto_onnxrt` [OnnxRuntime] and [OnnxSession] API. Produces L2-normalised
/// float32 embeddings suitable for cosine similarity search.
///
/// ## Model identity
///
/// [modelId] returns the stable [ModelSpec.id] of the loaded model (e.g.
/// `bge-small-en-v1.5`). This should be persisted alongside the vector index
/// so that a model change can be detected and the index rebuilt.
///
/// ## Loading with download-on-demand (preferred)
///
/// Supply a [cacheDir] (and optionally a [ModelSpec] via [spec]) to fetch the
/// model on first use via [ModelDownloader]. If the model files are already
/// cached and their SHA-256 checksums match, they are used immediately.
/// Otherwise [ModelDownloader] fetches the files before opening the ORT
/// session:
///
/// ```dart
/// final spec = ModelCatalog.lookup('bge-small-en-v1.5');
/// final model = await OnnxEmbeddingModel.load(
///   spec: spec,
///   cacheDir: '/path/to/cache',
///   onProgress: (received, total) {
///     stderr.writeln('Downloading: $received / $total bytes');
///   },
/// );
/// ```
///
/// ## Loading from an explicit path
///
/// The [modelPath] parameter loads a model from a specific filesystem path,
/// bypassing the catalog and downloader. Specifying [modelPath] without [spec]
/// uses [ModelCatalog.defaultModelId] for the identity.
///
/// ```dart
/// final model = await OnnxEmbeddingModel.load(
///   modelPath: '/path/to/bge_small.onnx',
/// );
/// ```
///
/// **Important:** Either [modelPath] or [cacheDir] must be supplied.
/// Calling [load] without either throws [ArgumentError] synchronously — there
/// is no bundled model asset path. See [ModelCatalog] and [ModelDownloader].
///
/// ## Lifecycle
///
/// [load] opens the native ORT session via [OnnxRuntime.load]. [embed] runs
/// synchronously on the calling isolate — do **not** call from the UI thread
/// in Flutter without isolate offloading. [dispose] releases native resources;
/// always call it (use `try/finally`).
///
/// ## Thread safety
///
/// ORT sessions are thread-affine. All [embed] and [dispose] calls must come
/// from the same isolate that called [load].
class OnnxEmbeddingModel implements EmbeddingModel {
  /// Internal constructor — use [load].
  OnnxEmbeddingModel._(
    this._runtime,
    this._session,
    this._tokenizer,
    this._spec,
  );

  final OnnxRuntime _runtime;
  final OnnxSession _session;
  final ModelTokenizer _tokenizer;

  /// The [ModelSpec] of the loaded model.
  ///
  /// Provides [modelId] and [dimensions] for the [EmbeddingModel] interface.
  final ModelSpec _spec;

  // ── EmbeddingModel interface ───────────────────────────────────────────────

  /// Stable identifier of the loaded model, matching a [ModelCatalog] entry.
  ///
  /// Should be persisted alongside the vector index so a later model swap can
  /// be detected and the index rebuilt. Example: `bge-small-en-v1.5`.
  @override
  String get modelId => _spec.id; // coverage:ignore-line

  /// Embedding vector length produced by this model.
  ///
  /// Sourced from `spec.meta['dimensions']`. This is the single source of
  /// truth for SQ8 byte lengths and score-path length guards.
  /// Example: 384 for BGE Small En v1.5.
  @override
  int get dimensions => _spec.meta['dimensions'] as int; // coverage:ignore-line

  // ── Factory ────────────────────────────────────────────────────────────────

  /// Loads an embedding model and returns an [OnnxEmbeddingModel].
  ///
  /// **Either [modelPath] or [cacheDir] must be supplied.** Calling [load]
  /// without either throws [ArgumentError] synchronously — there is no default
  /// asset path or bundled model. Use [cacheDir] to download the model on
  /// demand (preferred), or [modelPath] to load from an explicit filesystem
  /// path. See [ModelCatalog] and [ModelDownloader].
  ///
  /// ## Download-on-demand path (preferred)
  ///
  /// When [cacheDir] is provided, [ModelDownloader] is invoked to ensure the
  /// model files are present and checksummed before opening the ORT session.
  /// Files already in the cache are reused without downloading. Pass [spec] to
  /// select a specific catalog model; if omitted, [ModelCatalog.defaultModelId]
  /// (`'bge-small-en-v1.5'`) is used.
  ///
  /// [onProgress] is forwarded to [ModelDownloader.ensure] and receives
  /// incremental download progress. It is not called when files are cached.
  ///
  /// ```dart
  /// final model = await OnnxEmbeddingModel.load(
  ///   cacheDir: '/path/to/cache',
  ///   onProgress: (received, total) => print('$received / $total'),
  /// );
  /// ```
  ///
  /// ## Explicit-path
  ///
  /// When [modelPath] is provided, the file at that path is loaded directly
  /// (no download, no checksum). The model identity is set to
  /// [ModelCatalog.defaultModelId] unless [spec] is also supplied.
  ///
  /// [tokenizer] overrides the word-segmentation step inside [BertTokenizer].
  /// Defaults to [RegExpTokenizer]. Supply `IcuTokenizer()` from
  /// `package:betto_icu` for superior Unicode coverage. Ignored for models
  /// whose `tokenizerFamily` is `'xlmr'` (word segmentation there is handled
  /// entirely by [XlmRobertaTokenizer]'s SentencePiece/Unigram pipeline, which
  /// has no equivalent seam).
  ///
  /// ## Tokenizer family selection
  ///
  /// The concrete [ModelTokenizer] implementation is chosen from
  /// `spec.meta['tokenizerFamily']`: `'bert'` loads [BertTokenizer],
  /// `'xlmr'` loads [XlmRobertaTokenizer]. This key must be present and
  /// recognised — an absent or unrecognised value throws [ArgumentError]
  /// rather than silently defaulting, so a future third tokenizer family
  /// can't be misresolved by accident.
  ///
  /// Throws [ArgumentError] if neither [modelPath] nor [cacheDir] is supplied,
  /// or if [spec]`.meta['tokenizerFamily']` is missing or unrecognised.
  /// Throws [UnsupportedError] if the model file does not exist on disk.
  /// Throws [Exception] if the ORT library cannot be loaded or the model is
  /// corrupt.
  static Future<OnnxEmbeddingModel> load({
    ModelSpec? spec,
    String? cacheDir,
    String? modelPath,
    Tokenizer? tokenizer,
    DownloadProgress? onProgress,
  }) async {
    // Guard: at least one of modelPath or cacheDir must be supplied.
    // This is a required-argument check: throw synchronously before any I/O
    // so callers get a fast, clear failure rather than a confusing downstream
    // error. The bundled LFS asset path has been removed — download-on-demand
    // (cacheDir) or an explicit modelPath is the only supported mechanism.
    if (modelPath == null && cacheDir == null) {
      throw ArgumentError(
        'Either modelPath or cacheDir must be supplied. '
        'Pass an explicit modelPath, or pass cacheDir (with an optional spec) '
        'to download the model on demand. '
        'See ModelCatalog and ModelDownloader.',
      );
    }

    // Resolve the model spec. When no spec is given, use the default model.
    // When a raw modelPath is supplied without a spec, we still need an id for
    // model identity tracking — use the default catalog ID.
    final resolvedSpec =
        spec ?? ModelCatalog.lookup(ModelCatalog.defaultModelId);

    // Validate tokenizerFamily synchronously, before any I/O — same
    // fail-fast rationale as the modelPath/cacheDir guard above. This is a
    // pure, fast check (no file or network access), so it is fully covered
    // by unit tests rather than requiring live model assets.
    final tokenizerFamily = _tokenizerFamily(resolvedSpec);

    final String resolvedModelPath;
    final String resolvedVocabPath;

    if (modelPath != null) {
      // Explicit path — bypass catalog and downloader. The second asset file
      // is named 'vocab.txt' regardless of tokenizer family for this path —
      // an explicit modelPath is a lower-level escape hatch than the
      // catalog/downloader path below, so it keeps the original, simpler
      // convention rather than branching on tokenizerFamily too.
      resolvedModelPath = modelPath;
      resolvedVocabPath = p.join(p.dirname(modelPath), 'vocab.txt');
    } else {
      // cacheDir != null is guaranteed by the guard above.
      // Download-on-demand path: let ModelDownloader ensure the files are
      // present and their checksums match before opening the ORT session.
      // The ModelCatalog allowlist gates which models may be downloaded.
      final downloader = ModelDownloader(allowlist: ModelCatalog());
      final resolved = await downloader.ensure(
        resolvedSpec,
        cacheDir: cacheDir!,
        onProgress: onProgress,
      );
      // File names in ResolvedModel.filePaths match the keys in ModelSpec.files:
      // 'onnx' → absolute path to the .onnx model, 'vocab' → the tokenizer
      // asset (vocab.txt for BERT-family models, tokenizer.json for
      // XLM-R-family models — the key name is a generic "second asset" slot,
      // not tied to any one file format).
      resolvedModelPath = resolved.filePaths['onnx']!;
      resolvedVocabPath = resolved.filePaths['vocab']!;
    }

    _assertFileExists(resolvedModelPath, 'model file');
    _assertFileExists(resolvedVocabPath, 'tokenizer asset');

    // coverage:ignore-start
    // The lines below require a live ORT native library and model assets.
    // They are tested through integration tests when model assets are present.
    final runtime = await OnnxRuntime.load();
    final session = runtime.createSessionFromFile(resolvedModelPath);
    // tokenizerFamily was already validated above (fail-fast, before I/O) —
    // this switch only selects which concrete loader to invoke.
    final ModelTokenizer tok = tokenizerFamily == 'xlmr'
        ? await XlmRobertaTokenizer.load(resolvedVocabPath)
        : await BertTokenizer.load(resolvedVocabPath, tokenizer: tokenizer);
    return OnnxEmbeddingModel._(runtime, session, tok, resolvedSpec);
    // coverage:ignore-end
  }

  /// Validates and returns `spec.meta['tokenizerFamily']`.
  ///
  /// Pure and synchronous — safe to call before any I/O, so a misconfigured
  /// [ModelSpec] fails fast with a clear message rather than surfacing as a
  /// confusing downstream error once the ORT session is already open.
  ///
  /// Throws [ArgumentError] if the key is missing or is not one of the
  /// recognised values (`'bert'`, `'xlmr'`) — deliberately not defaulted, so
  /// a future third tokenizer family can't be silently misresolved by a
  /// [ModelSpec] that forgot to set it.
  static String _tokenizerFamily(ModelSpec spec) {
    final family = spec.meta['tokenizerFamily'];
    if (family != 'bert' && family != 'xlmr') {
      throw ArgumentError(
        "ModelSpec '${spec.id}' has meta['tokenizerFamily'] = "
        "${family == null ? 'null (missing)' : "'$family'"}, but only "
        "'bert' and 'xlmr' are recognised. Add a valid tokenizerFamily "
        'entry to the ModelSpec.meta map.',
      );
    }
    return family as String;
  }

  /// Prepends the `kind`-appropriate prefix from `spec.meta` to [text], if
  /// one is configured.
  ///
  /// Looks up `spec.meta['queryPrefix']` for [EmbeddingKind.query] or
  /// `spec.meta['documentPrefix']` for [EmbeddingKind.document]. If the
  /// corresponding key is absent, [text] is returned unchanged — this is a
  /// deliberate no-op default so models that don't need a prefix (e.g.
  /// `bge-small-en-v1.5`) are byte-for-byte unaffected by [kind].
  ///
  /// Exposed (rather than kept private) so this pure, spec-driven logic can
  /// be unit-tested directly without needing a live ORT session — see this
  /// package's coverage notes for why [embed] itself is `coverage:ignore`d.
  @visibleForTesting
  static String applyPrefix(String text, EmbeddingKind kind, ModelSpec spec) {
    final key = kind == EmbeddingKind.query ? 'queryPrefix' : 'documentPrefix';
    final prefix = spec.meta[key];
    return prefix is String ? '$prefix$text' : text;
  }

  // ── EmbeddingModel.embed ──────────────────────────────────────────────────

  /// Embeds [text] into an L2-normalised float32 vector of [dimensions] elements.
  ///
  /// Runs synchronously on the calling isolate. For large batches or UI
  /// applications, wrap calls in [Isolate.run] — but note that ORT sessions
  /// are thread-affine, so the session must be created inside the same isolate
  /// that calls [embed].
  ///
  /// An empty or whitespace-only [text] produces a `[CLS][SEP]`-only
  /// embedding (two real tokens) and returns `truncated = false`.
  ///
  /// [kind] selects [spec.meta]'s `'queryPrefix'` (for
  /// [EmbeddingKind.query]) or `'documentPrefix'` (for
  /// [EmbeddingKind.document], the default) and prepends it to [text] before
  /// tokenisation — see [applyPrefix]. Models with neither key (e.g.
  /// `bge-small-en-v1.5`) are unaffected: the prepend is a no-op, so
  /// behaviour is byte-for-byte unchanged regardless of [kind].
  ///
  /// Returns `(embedding, truncated)`:
  /// - `embedding` — [dimensions]-element [Float32List] with unit L2 norm.
  /// - `truncated` — `true` if [text] (after prefixing) exceeded the usable
  ///   token budget and was silently cut before embedding.
  // coverage:ignore-start
  // This entire method requires a live ORT session to exercise meaningfully
  // — covered by integration tests with model assets, not make coverage.
  // (applyPrefix itself, the pure prefix-selection logic embed() delegates
  // to below, is unit-tested directly without needing a live session — see
  // its own doc comment and test/onnx_embedding_model_test.dart.)
  @override
  Future<(Float32List, bool)> embed(
    String text, {
    EmbeddingKind kind = EmbeddingKind.document,
  }) async {
    final prefixedText = applyPrefix(text, kind, _spec);
    final tokens = _tokenizer.encode(prefixedText);
    final seqLen = tokens.inputIds.length;
    final hiddenDim = dimensions; // sourced from spec.meta['dimensions']

    // Build int64 input tensors shaped [1, seqLen].
    // The BGE model requires three inputs: input_ids, attention_mask,
    // and token_type_ids — all shaped [1, seqLen] with int64 elements.
    final shape = [1, seqLen];
    final inputIds = OnnxTensor.fromInt64(shape, tokens.inputIds);
    final attentionMask = OnnxTensor.fromInt64(shape, tokens.attentionMask);
    final tokenTypeIds = OnnxTensor.fromInt64(shape, tokens.tokenTypeIds);

    // Run ONNX inference. The output 'last_hidden_state' has shape
    // [1, seqLen, hiddenDim]. We rely on OnnxSession.run() populating
    // the shape from the native OrtValue via the output-shape-readback
    // slots (31/32/33) added in the generic betto_onnxrt API.
    final outputs = _session.run(
      inputs: {
        'input_ids': inputIds,
        'attention_mask': attentionMask,
        'token_type_ids': tokenTypeIds,
      },
      outputNames: ['last_hidden_state'],
    );
    final outputTensor = outputs.first;

    // Extract the flat float32 output. The tensor shape is [1, seqLen, D].
    // asFloat32() throws StateError if the output element type is not float32,
    // which would indicate a mismatched model.
    final raw = outputTensor.asFloat32().toList();

    // Mean-pool over non-padding token positions, then L2-normalise.
    final pooled = meanPool(
      raw,
      tokens.attentionMask.toList(),
      seqLen: seqLen,
      hiddenDim: hiddenDim,
    );
    final embedding = l2Normalize(pooled);

    return (embedding, tokens.truncated);
    // coverage:ignore-end
  }

  /// Releases the native ORT session and runtime resources.
  ///
  /// Must be called exactly once when the model is no longer needed.
  /// After [dispose], [embed] must not be called.
  @override
  void dispose() {
    // coverage:ignore-start
    // Requires a live ORT session — covered by integration tests with model assets.
    _session.dispose();
    _runtime.dispose();
    // coverage:ignore-end
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  static void _assertFileExists(String path, String label) {
    if (!File(path).existsSync()) {
      throw UnsupportedError(
        '$label not found at: $path\n'
        'Ensure model assets are present or configure a cacheDir for '
        'download-on-demand. See ModelCatalog and ModelDownloader.',
      );
    }
  }
}
