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

import 'dart:typed_data';

/// Distinguishes indexing-time ("document") text from query-time ("query")
/// text passed to [EmbeddingModel.embed].
///
/// Some models (e.g. `multilingual-e5-small`) require a fixed textual prefix
/// (`"passage: "` / `"query: "`) depending on which side of retrieval the
/// text is on — encoding a query the same way as a document degrades
/// retrieval quality for those models. [EmbeddingKind] lets a caller state
/// which side it's on without needing to know whether the loaded model
/// actually requires a prefix; models that don't need one (e.g.
/// `bge-small-en-v1.5`) simply ignore it.
enum EmbeddingKind {
  /// Text being indexed (inserted or updated) into a vector index.
  document,

  /// Text used to query a vector index for nearest neighbours.
  query,
}

/// Abstract interface for text-to-vector embedding models.
///
/// Allows a consuming database or application to accept an embedding model
/// without taking a dependency on the FFI-heavy `betto_inferencing` package.
/// The concrete implementation (`OnnxEmbeddingModel`) lives in this package
/// and implements this interface.
///
/// ## Usage
///
/// Supply an [EmbeddingModel] to your database's open call when configuring
/// vector indexes:
///
/// ```dart
/// // Load the model using the concrete implementation in this package.
/// final model = await OnnxEmbeddingModel.load(
///   cacheDir: '/path/to/cache',
/// );
/// // Pass it to your database or retrieval pipeline.
/// // await myDatabase.open(embeddingModel: model);
/// ```
///
/// ## Contract
///
/// - [embed] is called once per document field during indexing and once per
///   query. Implementations must be safe to call from the main isolate.
/// - The returned [Float32List] must have exactly [dimensions] elements.
/// - If [text] is longer than the model's context window, the implementation
///   truncates and sets `truncated = true` in the returned record.
/// - Implementations must not throw on empty [text]. Behaviour for empty
///   input is otherwise implementation-defined — for example,
///   `OnnxEmbeddingModel` produces a real `[CLS][SEP]`-only embedding (not a
///   zero vector), with `truncated = false`. A model with a mandatory
///   [EmbeddingKind] prefix (e.g. `multilingual-e5-small`'s `"passage: "` /
///   `"query: "`) never actually tokenises truly empty content, since the
///   prefix is prepended before empty [text] is checked.
abstract interface class EmbeddingModel {
  /// Stable identifier of the model that produced these embeddings.
  ///
  /// Matches a `ModelCatalog` entry id (e.g. `bge-small-en-v1.5`). Persisted
  /// alongside index data so a later model swap can be detected and the index
  /// rebuilt.
  ///
  /// Must be non-empty and stable across process restarts. Must not change
  /// after the model is loaded.
  String get modelId;

  /// Embedding vector length produced by this model.
  ///
  /// The single source of truth for SQ8 byte lengths and score-path length
  /// guards in the consuming retrieval layer. For example, 384 for BGE Small
  /// En v1.5 and 1024 for BGE-M3.
  ///
  /// Must equal the length of the `embedding` field in every record returned by [embed].
  int get dimensions;

  /// Embeds [text] into a dense float vector.
  ///
  /// [kind] states whether [text] is being indexed ([EmbeddingKind.document],
  /// the default) or is a search query ([EmbeddingKind.query]). Models that
  /// require different handling for the two cases (e.g. a mandatory
  /// `"passage: "`/`"query: "` prefix) branch on [kind] internally; models
  /// that don't need this distinction ignore it entirely, so passing the
  /// default is always safe for those models. Callers that know which case
  /// they're in (indexing vs. querying) should always pass the matching
  /// [kind] explicitly — silently treating every call as [EmbeddingKind.document]
  /// degrades retrieval quality for models that do use the distinction,
  /// without erroring.
  ///
  /// Returns a record `(embedding, truncated)` where:
  /// - `embedding` is the float32 embedding vector with exactly [dimensions]
  ///   elements.
  /// - `truncated` is `true` if [text] exceeded the model's context window and
  ///   was silently truncated before embedding.
  Future<(Float32List embedding, bool truncated)> embed(
    String text, {
    EmbeddingKind kind = EmbeddingKind.document,
  });

  /// Releases any native resources held by this model.
  ///
  /// Called by the consuming database after all other cleanup. Implementations
  /// backed by native libraries (e.g. ONNX Runtime) must release their session
  /// handle here. Pure-Dart implementations may leave this as a no-op.
  ///
  /// After [dispose] is called, [embed] must not be called.
  void dispose();
}
