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

import 'package:betto_inferencing/betto_inferencing.dart';
import 'package:test/test.dart';

void main() {
  group('ModelSpec (via ModelCatalog)', () {
    test('BGE Small En v1.5 spec has correct dimensions in meta', () {
      final spec = ModelCatalog.lookup('bge-small-en-v1.5');
      expect(spec.id, equals('bge-small-en-v1.5'));
      // Dimensions are stored in spec.meta['dimensions'] — not as a direct
      // field — because betto_onnxrt uses a generic ModelSpec shape.
      expect(spec.meta['dimensions'], equals(384));
    });

    test('BGE Small En v1.5 has non-empty URLs and checksums', () {
      final spec = ModelCatalog.lookup('bge-small-en-v1.5');
      expect(spec.files['onnx']!.url.toString(), isNotEmpty);
      expect(spec.files['vocab']!.url.toString(), isNotEmpty);
      expect(spec.files['onnx']!.sha256, isNotEmpty);
      expect(spec.files['vocab']!.sha256, isNotEmpty);
    });

    // Regression: checksums were placeholder values (repeating 4e4e… pattern)
    // that passed the isNotEmpty check above but caused download verification
    // to fail at runtime. Pin the exact known-good values so any accidental
    // reversion or upstream change is caught at test time, not at runtime.
    test('BGE Small En v1.5 onnx sha256 matches known-good value', () {
      final spec = ModelCatalog.lookup('bge-small-en-v1.5');
      expect(
        spec.files['onnx']!.sha256,
        equals(
          '828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35',
        ),
      );
    });

    test('BGE Small En v1.5 vocab sha256 matches known-good value', () {
      final spec = ModelCatalog.lookup('bge-small-en-v1.5');
      expect(
        spec.files['vocab']!.sha256,
        equals(
          '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
        ),
      );
    });

    test('BGE Small En v1.5 onnx URL points to HuggingFace', () {
      final spec = ModelCatalog.lookup('bge-small-en-v1.5');
      expect(spec.files['onnx']!.url.toString(), contains('huggingface.co'));
    });

    test('BGE Small En v1.5 has onnx and vocab file entries', () {
      final spec = ModelCatalog.lookup('bge-small-en-v1.5');
      expect(spec.files.containsKey('onnx'), isTrue);
      expect(spec.files.containsKey('vocab'), isTrue);
    });

    test('BGE Small En v1.5 tokenizerFamily is bert', () {
      final spec = ModelCatalog.lookup('bge-small-en-v1.5');
      expect(spec.meta['tokenizerFamily'], equals('bert'));
    });

    test('BGE Small En v1.5 has no queryPrefix/documentPrefix', () {
      // No passage/query prefix convention for this model — absence is
      // meaningful (OnnxEmbeddingModel.embed()'s prefix step no-ops).
      final spec = ModelCatalog.lookup('bge-small-en-v1.5');
      expect(spec.meta.containsKey('queryPrefix'), isFalse);
      expect(spec.meta.containsKey('documentPrefix'), isFalse);
    });
  });

  group('ModelSpec (via ModelCatalog) — multilingual-e5-small', () {
    test('spec has correct dimensions in meta', () {
      final spec = ModelCatalog.lookup('multilingual-e5-small');
      expect(spec.id, equals('multilingual-e5-small'));
      expect(spec.meta['dimensions'], equals(384));
    });

    test('tokenizerFamily is xlmr', () {
      final spec = ModelCatalog.lookup('multilingual-e5-small');
      expect(spec.meta['tokenizerFamily'], equals('xlmr'));
    });

    test('queryPrefix and documentPrefix are set per the E5 model card', () {
      final spec = ModelCatalog.lookup('multilingual-e5-small');
      expect(spec.meta['queryPrefix'], equals('query: '));
      expect(spec.meta['documentPrefix'], equals('passage: '));
    });

    test('has non-empty URLs and checksums', () {
      final spec = ModelCatalog.lookup('multilingual-e5-small');
      expect(spec.files['onnx']!.url.toString(), isNotEmpty);
      expect(spec.files['vocab']!.url.toString(), isNotEmpty);
      expect(spec.files['onnx']!.sha256, isNotEmpty);
      expect(spec.files['vocab']!.sha256, isNotEmpty);
    });

    // Regression guard, same rationale as BGE Small En v1.5's pinned-checksum
    // test above: pin the exact known-good values (computed by
    // tool/register_model.dart from a real download) so an accidental
    // reversion to a placeholder or an unnoticed upstream change is caught at
    // test time, not at runtime.
    test('onnx sha256 matches known-good value', () {
      final spec = ModelCatalog.lookup('multilingual-e5-small');
      expect(
        spec.files['onnx']!.sha256,
        equals(
          'ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665',
        ),
      );
    });

    test('tokenizer.json sha256 matches known-good value', () {
      final spec = ModelCatalog.lookup('multilingual-e5-small');
      expect(
        spec.files['vocab']!.sha256,
        equals(
          '0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39',
        ),
      );
    });

    test('onnx URL points to HuggingFace intfloat/multilingual-e5-small', () {
      final spec = ModelCatalog.lookup('multilingual-e5-small');
      expect(
        spec.files['onnx']!.url.toString(),
        allOf(
          contains('huggingface.co'),
          contains('intfloat/multilingual-e5-small'),
        ),
      );
    });

    test('is registered and validated', () {
      expect(ModelCatalog.isKnown('multilingual-e5-small'), isTrue);
      // lookup() would throw UnsupportedError if unvalidated — reaching this
      // line without throwing is the assertion.
      ModelCatalog.lookup('multilingual-e5-small');
    });
  });

  group('ModelCatalog.lookup', () {
    test('returns the correct spec for a known validated model', () {
      final spec = ModelCatalog.lookup('bge-small-en-v1.5');
      expect(spec.id, equals('bge-small-en-v1.5'));
      expect(spec.meta['dimensions'], equals(384));
    });

    test('throws ArgumentError for an unknown model ID', () {
      expect(
        () => ModelCatalog.lookup('unknown-model-v99'),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            contains('Unknown embedding model ID'),
          ),
        ),
      );
    });

    test('error message for unknown model lists registered IDs', () {
      expect(
        () => ModelCatalog.lookup('unknown-model'),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message as String,
            'message',
            allOf(contains('bge-small-en-v1.5'), contains('placeholder-model')),
          ),
        ),
      );
    });

    test('throws UnsupportedError for a registered but unvalidated model', () {
      // placeholder-model is a permanent test fixture — always registered,
      // never validated. See ModelCatalog's doc comment.
      expect(
        () => ModelCatalog.lookup('placeholder-model'),
        throwsA(
          isA<UnsupportedError>().having(
            (e) => e.message,
            'message',
            contains('not yet been validated'),
          ),
        ),
      );
    });

    test('error message for unvalidated model mentions the model ID', () {
      expect(
        () => ModelCatalog.lookup('placeholder-model'),
        throwsA(
          isA<UnsupportedError>().having(
            (e) => e.message,
            'message',
            contains('placeholder-model'),
          ),
        ),
      );
    });
  });

  group('ModelCatalog.isKnown', () {
    test('returns true for a registered validated model', () {
      expect(ModelCatalog.isKnown('bge-small-en-v1.5'), isTrue);
    });

    test('returns true for a registered but unvalidated model', () {
      // isKnown does not check validation state — it only checks registration.
      expect(ModelCatalog.isKnown('placeholder-model'), isTrue);
    });

    test('returns false for an unknown model ID', () {
      expect(ModelCatalog.isKnown('nonexistent-model'), isFalse);
    });
  });

  group('ModelCatalog.all', () {
    test(
      'contains at least three entries (BGE Small En, multilingual-e5-small, '
      'placeholder-model)',
      () {
        final all = ModelCatalog.all.toList();
        expect(all.length, greaterThanOrEqualTo(3));
      },
    );

    test('contains BGE Small En v1.5', () {
      expect(ModelCatalog.all.any((s) => s.id == 'bge-small-en-v1.5'), isTrue);
    });

    test('contains multilingual-e5-small', () {
      expect(
        ModelCatalog.all.any((s) => s.id == 'multilingual-e5-small'),
        isTrue,
      );
    });

    test('contains placeholder-model (test fixture, unvalidated)', () {
      expect(ModelCatalog.all.any((s) => s.id == 'placeholder-model'), isTrue);
    });
  });

  group('ModelCatalog.defaultModelId', () {
    test('default model ID is bge-small-en-v1.5', () {
      expect(ModelCatalog.defaultModelId, equals('bge-small-en-v1.5'));
    });

    test('default model is validated', () {
      final spec = ModelCatalog.lookup(ModelCatalog.defaultModelId);
      expect(spec.id, equals(ModelCatalog.defaultModelId));
    });
  });

  group('ModelCatalog as AllowlistProvider', () {
    test('isAllowed returns true for a registered model', () {
      const catalog = ModelCatalog();
      final spec = ModelCatalog.lookup('bge-small-en-v1.5');
      expect(catalog.isAllowed(spec), isTrue);
    });

    test('isAllowed returns true for a registered but unvalidated model', () {
      const catalog = ModelCatalog();
      // placeholder-model is registered but unvalidated — isAllowed only
      // checks registration, not validation status.
      final placeholderSpec = ModelCatalog.all.firstWhere(
        (s) => s.id == 'placeholder-model',
      );
      expect(catalog.isAllowed(placeholderSpec), isTrue);
    });

    test('isAllowed returns false for an unregistered model', () {
      const catalog = ModelCatalog();
      const unknownSpec = ModelSpec(id: 'unknown-model', files: {});
      expect(catalog.isAllowed(unknownSpec), isFalse);
    });
  });
}
