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
// Covers only the offline, vocab-independent portion of XlmRobertaTokenizer
// (the charsmap-substitution + whitespace-collapse + Metaspace-escaping
// pipeline exposed as `normalizeForTokenization`). `load()` and the
// vocab-dependent tail of `encode()` require the real ~17 MB
// multilingual-e5-small tokenizer.json and are covered by the network-gated
// integration test in `integration_test_app/integration_test/`, not here —
// see this file's and that test's doc comments, and
// `plan_0_06_wi11_xlmr_tokenizer.md`'s Phase 1 "coverage strategy" section.

@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';

import 'package:betto_inferencing/src/charsmap_trie.dart';
import 'package:betto_inferencing/src/xlmr_tokenizer.dart';
import 'package:test/test.dart';

/// Path to the small, committed charsmap fixture shared with
/// `charsmap_trie_test.dart` — see that file's doc comment for provenance.
String get _charsmapFixturePath {
  return '${Directory.current.path}/test/fixtures/xlmr_precompiled_charsmap.b64';
}

late CharsmapTrie _trie;

void main() {
  setUpAll(() {
    final b64 = File(_charsmapFixturePath).readAsStringSync();
    _trie = CharsmapTrie.parse(base64.decode(b64));
  });

  group('XlmRobertaTokenizer.normalizeForTokenization', () {
    test('plain text gets a leading dummy-prefix space and all spaces become '
        'the metaspace symbol (▁)', () {
      final result = XlmRobertaTokenizer.normalizeForTokenization(
        _trie,
        'hello world',
      );
      expect(result, equals('▁hello▁world'));
    });

    test('runs of two or more spaces collapse to one before metaspacing', () {
      final result = XlmRobertaTokenizer.normalizeForTokenization(
        _trie,
        'a   b',
      );
      expect(result, equals('▁a▁b'));
    });

    test('text that already has a leading space is not given a second one', () {
      final result = XlmRobertaTokenizer.normalizeForTokenization(
        _trie,
        ' Hello',
      );
      expect(result, equals('▁Hello'));
    });

    test('charsmap substitution (fullwidth folding) runs before whitespace '
        'collapse and metaspacing', () {
      // Fullwidth "Hello   World" (3 plain ASCII spaces in between, which
      // the charsmap substitution step leaves untouched) must fold to
      // ASCII first, then have its space run collapsed, then be
      // metaspaced -- exercising all three pipeline steps together in the
      // order XlmRobertaTokenizer.encode() applies them.
      final result = XlmRobertaTokenizer.normalizeForTokenization(
        _trie,
        'Ｈｅｌｌｏ   Ｗｏｒｌｄ',
      );
      expect(result, equals('▁Hello▁World'));
    });

    test('empty string is left empty, not given a dummy-prefix metaspace '
        'symbol', () {
      // Found via WI-4's broader parity corpus
      // (test/fixtures/xlmr_parity_corpus.json's edge_empty entry): real
      // AutoTokenizer output for "" is just [<s>, </s>], with no content
      // token -- unconditionally adding the dummy prefix here would turn ""
      // into "▁", which the real vocabulary tokenizes as a standalone piece
      // (id 6), producing a spurious extra token. See `_metaspace`'s doc
      // comment for the full explanation.
      final result = XlmRobertaTokenizer.normalizeForTokenization(_trie, '');
      expect(result, isEmpty);
    });
  });

  group('XlmRobertaTokenizer.load - error handling (offline)', () {
    test('throws FormatException for tokenizer.json with no Precompiled '
        'normalizer entry', () async {
      // A tokenizer.json shaped enough to reach the charsmap-extraction
      // step, but whose normalizer section declares no Precompiled entry
      // -- exercises the guard without needing the real 17 MB file with a
      // full 250k-entry vocab.
      final tmpDir = await Directory.systemTemp.createTemp(
        'xlmr_tokenizer_test',
      );
      final path = '${tmpDir.path}/tokenizer.json';
      await File(path).writeAsString(
        jsonEncode({
          'normalizer': {
            'type': 'Sequence',
            'normalizers': [
              {'type': 'Replace', 'content': ' '},
            ],
          },
          'model': {'type': 'Unigram', 'vocab': <List<Object?>>[]},
        }),
      );
      try {
        await expectLater(
          XlmRobertaTokenizer.load(path),
          throwsFormatException,
        );
      } finally {
        await tmpDir.delete(recursive: true);
      }
    });

    test('extracts and parses a Precompiled charsmap entry when present, even '
        'when the rest of the file is not a usable tokenizer.json', () async {
      // Exercises _extractCharsmapTrie's success path (a valid Precompiled
      // entry is found and its bytes parse cleanly) using the same minimal
      // single-unit trie as CharsmapTrie's own malformed-input tests --
      // this only needs a well-formed charsmap, not a real vocabulary. The
      // 'model' section is deliberately omitted so that the subsequent
      // HuggingFaceTokenizerLoader.fromJsonString call (inside load()'s
      // coverage:ignore block) fails fast rather than requiring a real
      // vocab -- irrelevant here since only _extractCharsmapTrie's
      // behaviour is under test.
      final minimalTrieBytes = base64.encode([4, 0, 0, 0, 0, 0, 0, 0]);
      final tmpDir = await Directory.systemTemp.createTemp(
        'xlmr_tokenizer_test',
      );
      final path = '${tmpDir.path}/tokenizer.json';
      await File(path).writeAsString(
        jsonEncode({
          'normalizer': {
            'type': 'Sequence',
            'normalizers': [
              {'type': 'Precompiled', 'precompiled_charsmap': minimalTrieBytes},
            ],
          },
        }),
      );
      try {
        // load() as a whole still throws (no 'model' section), but only
        // after successfully building the CharsmapTrie -- confirmed by
        // the FormatException's message not mentioning "Precompiled".
        await expectLater(
          XlmRobertaTokenizer.load(path),
          throwsA(
            isA<FormatException>().having(
              (e) => e.message,
              'message',
              isNot(contains('Precompiled')),
            ),
          ),
        );
      } finally {
        await tmpDir.delete(recursive: true);
      }
    });
  });
}
