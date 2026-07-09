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
// Covers ModelTokenizer itself. Per-implementation conformance
// (BertTokenizer, XlmRobertaTokenizer) is covered in each tokenizer's own
// test file -- see bert_tokenizer_test.dart's "ModelTokenizer conformance"
// group (a live, runtime golden-output check) and xlmr_tokenizer_test.dart's
// (a compile-time-only check, since a live XlmRobertaTokenizer instance
// requires the real ~17 MB tokenizer.json).

@TestOn('vm')
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:betto_inferencing/betto_inferencing.dart';
import 'package:test/test.dart';

/// Path to the synthetic fixture vocab shared with bert_tokenizer_test.dart.
String get _vocabPath => '${Directory.current.path}/test/fixtures/vocab.txt';

void main() {
  group('ModelTokenizer', () {
    test('a hand-written implementation satisfies the interface contract', () {
      // A minimal, independent ModelTokenizer implementation -- confirms the
      // interface itself (encode(String) -> TokenizerOutput) is implementable
      // by third-party code, not just this package's own two tokenizers.
      const impl = _FixedOutputTokenizer();
      final ModelTokenizer asModelTokenizer = impl;
      final output = asModelTokenizer.encode('anything');
      expect(output.inputIds, equals(Int64List.fromList([1, 2, 3])));
      expect(output.truncated, isFalse);
    });

    test('BertTokenizer and a hand-written ModelTokenizer are interchangeable '
        'through the same interface reference', () async {
      final bertTokenizer = await BertTokenizer.load(_vocabPath);
      final tokenizers = <ModelTokenizer>[
        bertTokenizer,
        const _FixedOutputTokenizer(),
      ];
      // Both must be callable uniformly through the ModelTokenizer
      // interface -- this is the entire point of the abstraction: a single
      // OnnxEmbeddingModel call site works regardless of which concrete
      // tokenizer family is loaded.
      for (final tokenizer in tokenizers) {
        expect(tokenizer.encode('hello'), isA<TokenizerOutput>());
      }
    });
  });
}

/// A minimal [ModelTokenizer] implementation independent of this package's
/// own tokenizers, proving the interface is implementable by third-party
/// code and not accidentally coupled to [BertTokenizer]'s or
/// [XlmRobertaTokenizer]'s internals.
final class _FixedOutputTokenizer implements ModelTokenizer {
  const _FixedOutputTokenizer();

  @override
  TokenizerOutput encode(String text) => TokenizerOutput(
    inputIds: Int64List.fromList([1, 2, 3]),
    attentionMask: Int64List.fromList([1, 1, 1]),
    tokenTypeIds: Int64List.fromList([0, 0, 0]),
    truncated: false,
  );
}
