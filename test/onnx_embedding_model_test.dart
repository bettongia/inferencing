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
// Covers the pure, offline logic inside OnnxEmbeddingModel that does not
// require a live ORT session or real model assets:
//   - applyPrefix: the queryPrefix/documentPrefix lookup driven by
//     EmbeddingKind (WI-4, Q3).
//   - tokenizerFamily validation inside load(): a synchronous, pre-I/O guard
//     (WI-4, Q2) -- see onnx_embedding_model.dart's `_tokenizerFamily`.
// The tokenizer-selection and embed() inference paths themselves require a
// live ORT session and real model assets and are covered by the network-
// gated integration test in integration_test_app/integration_test/.

@TestOn('vm')
library;

import 'package:betto_inferencing/betto_inferencing.dart';
import 'package:test/test.dart';

void main() {
  group('OnnxEmbeddingModel.applyPrefix', () {
    test('prepends documentPrefix for EmbeddingKind.document when set', () {
      const spec = ModelSpec(
        id: 'test-model',
        files: {},
        meta: {'documentPrefix': 'passage: ', 'queryPrefix': 'query: '},
      );
      expect(
        OnnxEmbeddingModel.applyPrefix('hello', EmbeddingKind.document, spec),
        equals('passage: hello'),
      );
    });

    test('prepends queryPrefix for EmbeddingKind.query when set', () {
      const spec = ModelSpec(
        id: 'test-model',
        files: {},
        meta: {'documentPrefix': 'passage: ', 'queryPrefix': 'query: '},
      );
      expect(
        OnnxEmbeddingModel.applyPrefix('hello', EmbeddingKind.query, spec),
        equals('query: hello'),
      );
    });

    test('is a no-op for both kinds when neither prefix key is present '
        '(bge-small-en-v1.5 behaviour)', () {
      const spec = ModelSpec(id: 'test-model', files: {});
      expect(
        OnnxEmbeddingModel.applyPrefix('hello', EmbeddingKind.document, spec),
        equals('hello'),
      );
      expect(
        OnnxEmbeddingModel.applyPrefix('hello', EmbeddingKind.query, spec),
        equals('hello'),
      );
    });

    test('is a no-op for document kind when only queryPrefix is set', () {
      const spec = ModelSpec(
        id: 'test-model',
        files: {},
        meta: {'queryPrefix': 'query: '},
      );
      expect(
        OnnxEmbeddingModel.applyPrefix('hello', EmbeddingKind.document, spec),
        equals('hello'),
      );
    });

    test('is a no-op for query kind when only documentPrefix is set', () {
      const spec = ModelSpec(
        id: 'test-model',
        files: {},
        meta: {'documentPrefix': 'passage: '},
      );
      expect(
        OnnxEmbeddingModel.applyPrefix('hello', EmbeddingKind.query, spec),
        equals('hello'),
      );
    });

    test('applies the prefix to empty text too', () {
      const spec = ModelSpec(
        id: 'test-model',
        files: {},
        meta: {'documentPrefix': 'passage: '},
      );
      expect(
        OnnxEmbeddingModel.applyPrefix('', EmbeddingKind.document, spec),
        equals('passage: '),
      );
    });
  });

  group('OnnxEmbeddingModel.load - tokenizerFamily validation', () {
    // These exercise a synchronous guard that runs before any file or
    // network I/O (see onnx_embedding_model.dart's _tokenizerFamily), so
    // they can be asserted without real model assets even though modelPath
    // points nowhere real.
    test(
      'throws ArgumentError when spec.meta has no tokenizerFamily key',
      () async {
        const spec = ModelSpec(id: 'no-family', files: {});
        await expectLater(
          OnnxEmbeddingModel.load(
            modelPath: '/nonexistent/model.onnx',
            spec: spec,
          ),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.message,
              'message',
              allOf(contains('tokenizerFamily'), contains('missing')),
            ),
          ),
        );
      },
    );

    test(
      'throws ArgumentError when tokenizerFamily is not "bert" or "xlmr"',
      () async {
        const spec = ModelSpec(
          id: 'unknown-family',
          files: {},
          meta: {'tokenizerFamily': 'not-a-real-family'},
        );
        await expectLater(
          OnnxEmbeddingModel.load(
            modelPath: '/nonexistent/model.onnx',
            spec: spec,
          ),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.message,
              'message',
              contains('not-a-real-family'),
            ),
          ),
        );
      },
    );

    test(
      'does not throw ArgumentError for a valid tokenizerFamily (fails later '
      'with UnsupportedError once the missing-file check runs instead)',
      () async {
        const spec = ModelSpec(
          id: 'valid-family',
          files: {},
          meta: {'tokenizerFamily': 'xlmr'},
        );
        // A valid tokenizerFamily should pass the guard above and instead
        // fail at the (later) missing-model-file check -- proving the
        // ArgumentError guard doesn't misfire on legitimate values.
        await expectLater(
          OnnxEmbeddingModel.load(
            modelPath: '/nonexistent/model.onnx',
            spec: spec,
          ),
          throwsA(isA<UnsupportedError>()),
        );
      },
    );
  });
}
