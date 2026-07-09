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
//
// Covers EmbeddingKind and the EmbeddingModel.embed() contract shape. The
// concrete OnnxEmbeddingModel's use of EmbeddingKind (prefix application) is
// covered in onnx_embedding_model_test.dart via the pure, offline
// OnnxEmbeddingModel.applyPrefix helper.

@TestOn('vm')
library;

import 'dart:typed_data';

import 'package:betto_inferencing/betto_inferencing.dart';
import 'package:test/test.dart';

void main() {
  group('EmbeddingKind', () {
    test('has exactly two values: document and query', () {
      expect(
        EmbeddingKind.values,
        equals([EmbeddingKind.document, EmbeddingKind.query]),
      );
    });

    test('document is distinct from query', () {
      expect(EmbeddingKind.document, isNot(equals(EmbeddingKind.query)));
    });
  });

  group('EmbeddingModel.embed contract', () {
    test(
      'kind defaults to EmbeddingKind.document when not passed explicitly',
      () async {
        final model = _RecordingEmbeddingModel();
        await model.embed('some text');
        expect(model.lastKind, equals(EmbeddingKind.document));
      },
    );

    test(
      'kind: EmbeddingKind.query is threaded through to the implementation',
      () async {
        final model = _RecordingEmbeddingModel();
        await model.embed('some text', kind: EmbeddingKind.query);
        expect(model.lastKind, equals(EmbeddingKind.query));
      },
    );
  });
}

/// A minimal [EmbeddingModel] that records the [EmbeddingKind] it was last
/// called with, so the interface's default-parameter behaviour (and that
/// callers can override it) can be asserted independent of any concrete
/// model implementation.
final class _RecordingEmbeddingModel implements EmbeddingModel {
  EmbeddingKind? lastKind;

  @override
  String get modelId => 'recording-test-model';

  @override
  int get dimensions => 3;

  @override
  Future<(Float32List, bool)> embed(
    String text, {
    EmbeddingKind kind = EmbeddingKind.document,
  }) async {
    lastKind = kind;
    return (Float32List.fromList([0, 0, 0]), false);
  }

  @override
  void dispose() {}
}
