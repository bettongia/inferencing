// Copyright 2026 The Authors
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

// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:betto_inferencing/betto_inferencing.dart';

/// Example demonstrating SQ8 scalar quantisation — compact storage of float32
/// embedding vectors as uint8 bytes.
///
/// [quantise] maps each L2-normalised float32 component from `[-1, 1]` to a
/// uint8 in `[0, 255]`. [dequantise] reverses the mapping. Together they shrink
/// a 384-dimensional BGE Small En v1.5 vector from 1 536 bytes (float32) to
/// 384 bytes (uint8) — a 4× reduction — with a bounded round-trip error of at
/// most `2 / 255 ≈ 0.00784` per element.
///
/// This compact format is what `betto_db` stores on disk for each indexed
/// document. Similarity scores computed from dequantised vectors are close
/// enough to float32 scores that ranking order is preserved.
///
/// Set BETTO_CACHE to a persistent directory to avoid re-downloading the model
/// on each run:
///   BETTO_CACHE=$HOME/.cache/betto_examples dart run example/sq8_quantisation.dart
Future<void> main() async {
  final cacheDir =
      Platform.environment['BETTO_CACHE'] ??
      Directory.systemTemp.createTempSync('betto_sq8_cache').path;

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
    const text =
        'Scalar quantisation compresses embeddings for efficient storage.';
    print('Input: "$text"\n');

    final (embedding, _) = await model.embed(text);

    // ── Storage comparison ───────────────────────────────────────────────────
    final quantised = quantise(embedding);

    final float32Bytes = embedding.lengthInBytes; // dims × 4
    final sq8Bytes = quantised.lengthInBytes; // dims × 1
    final savingPct = (sq8Bytes * 100 / float32Bytes).round();

    print('Storage:');
    print('  Float32 : $float32Bytes bytes  (${embedding.length} × 4)');
    print('  SQ8     : $sq8Bytes bytes  (${quantised.length} × 1)');
    print(
      '  SQ8 uses $savingPct% of float32 storage (${float32Bytes ~/ sq8Bytes}× smaller)\n',
    );

    // ── Reconstruction error ─────────────────────────────────────────────────
    // dequantise() maps each uint8 back to float32. The round-trip error is
    // bounded by one quantisation step: 2.0 / 255 ≈ 0.00784 per element.
    final reconstructed = dequantise(quantised);

    var maxErr = 0.0;
    var sumErr = 0.0;
    for (var i = 0; i < embedding.length; i++) {
      final err = (embedding[i] - reconstructed[i]).abs();
      sumErr += err;
      maxErr = max(maxErr, err);
    }
    final meanErr = sumErr / embedding.length;
    const theoreticalMax = 2.0 / 255;

    print('Reconstruction error (per element):');
    print('  Max  : ${maxErr.toStringAsFixed(6)}');
    print('  Mean : ${meanErr.toStringAsFixed(6)}');
    print(
      '  Theoretical max : ${theoreticalMax.toStringAsFixed(6)} (= 2 / 255)\n',
    );

    // ── Similarity preservation ──────────────────────────────────────────────
    // Cosine similarity is the dot product of two L2-normalised vectors. We
    // compare the exact float32 score against the score computed from the
    // dequantised vectors to verify that ranking order is preserved.
    double dot(Float32List a, Float32List b) {
      var s = 0.0;
      for (var i = 0; i < a.length; i++) {
        s += a[i] * b[i];
      }
      return s;
    }

    // Embed a second text to compare against.
    const text2 = 'Lossy compression of high-dimensional vectors.';
    final (embedding2, _) = await model.embed(text2);
    final quantised2 = quantise(embedding2);
    final reconstructed2 = dequantise(quantised2);

    final exactScore = dot(embedding, embedding2);
    final sq8Score = dot(reconstructed, reconstructed2);

    print('Similarity: "$text"');
    print('        vs: "$text2"');
    print('  Float32 score : ${exactScore.toStringAsFixed(6)}');
    print('  SQ8 score     : ${sq8Score.toStringAsFixed(6)}');
    print(
      '  Delta         : ${(exactScore - sq8Score).abs().toStringAsFixed(6)}',
    );
  } finally {
    model.dispose();
  }
}
