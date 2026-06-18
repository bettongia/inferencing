// Copyright 2026 The Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:typed_data';

import 'package:betto_inferencing/betto_inferencing.dart';

/// Example demonstrating semantic similarity ranking with [OnnxEmbeddingModel].
///
/// Embeds a natural-language query alongside a set of candidate documents and
/// ranks the candidates by cosine similarity. Because [OnnxEmbeddingModel.embed]
/// returns L2-normalised vectors, cosine similarity is equivalent to the dot
/// product of the two vectors.
///
/// Set BETTO_CACHE to a persistent directory to avoid re-downloading the model
/// on each run:
///   BETTO_CACHE=$HOME/.cache/betto_examples dart run example/embed_and_compare.dart
Future<void> main() async {
  final cacheDir =
      Platform.environment['BETTO_CACHE'] ??
      Directory.systemTemp.createTempSync('betto_embed_cache').path;

  print('Loading model (cache: $cacheDir)…');

  final model = await OnnxEmbeddingModel.load(
    cacheDir: cacheDir,
    onProgress: (received, total) {
      final pct = total > 0 ? (received * 100 ~/ total) : 0;
      stdout.write('\r  Downloading: $pct%   ');
    },
  );
  stdout.writeln();

  try {
    print('Model : ${model.modelId}');
    print('Dims  : ${model.dimensions}\n');

    // ── Embed the query ──────────────────────────────────────────────────────
    // In a retrieval pipeline the query is embedded at search time and the
    // document vectors are pre-computed and stored (see sq8_quantisation.dart
    // for the compact storage format).
    const query = 'How does sentence embedding work?';
    final (queryVec, queryTruncated) = await model.embed(query);
    if (queryTruncated) print('  [warning] query was truncated');
    print('Query: "$query"\n');

    // ── Embed candidate documents ────────────────────────────────────────────
    final candidates = [
      'Sentence embeddings encode text as dense vectors for semantic search.',
      'The weather forecast predicts rain this weekend.',
      'Mean pooling over BERT hidden states produces sentence-level vectors.',
      'Neural networks learn distributed representations of language.',
      'A quick recipe for baking sourdough bread at home.',
    ];

    // cosine similarity = dot product for L2-normalised vectors.
    double cosine(Float32List a, Float32List b) {
      var dot = 0.0;
      for (var i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
      }
      return dot;
    }

    final scored = <({double score, String text, bool truncated})>[];
    for (final doc in candidates) {
      final (vec, truncated) = await model.embed(doc);
      scored.add((
        score: cosine(queryVec, vec),
        text: doc,
        truncated: truncated,
      ));
    }

    // ── Rank by descending similarity ────────────────────────────────────────
    scored.sort((a, b) => b.score.compareTo(a.score));

    print('Ranked by similarity to query:');
    for (var i = 0; i < scored.length; i++) {
      final r = scored[i];
      final flag = r.truncated ? ' [truncated]' : '';
      print('  ${i + 1}. (${r.score.toStringAsFixed(4)}) ${r.text}$flag');
    }
    print('');

    // ── Query vs. itself ─────────────────────────────────────────────────────
    // A vector's cosine similarity with itself is 1.0 for perfectly
    // L2-normalised vectors. Verifies the embedding is well-formed.
    final selfScore = cosine(queryVec, queryVec);
    print('Self-similarity of query vector: ${selfScore.toStringAsFixed(6)}');
  } finally {
    // Always dispose — releases the native ORT session.
    model.dispose();
  }
}
