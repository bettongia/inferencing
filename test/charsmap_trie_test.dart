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

@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:betto_inferencing/src/charsmap_trie.dart';
import 'package:test/test.dart';

/// Path to the real `multilingual-e5-small` `precompiled_charsmap`, base64
/// encoded, extracted from its `tokenizer.json` (see
/// `plan_0_06_wi11_xlmr_tokenizer.md`, Phase 1 fixture strategy). Only the
/// charsmap itself (~317 KB base64 / ~237 KB raw) is committed — not the
/// full 17 MB `tokenizer.json`, which is 98%+ an unused-for-this-purpose
/// 250k-entry vocab table.
String get _charsmapFixturePath {
  return '${Directory.current.path}/test/fixtures/xlmr_precompiled_charsmap.b64';
}

late CharsmapTrie _trie;

void main() {
  setUpAll(() {
    final b64 = File(_charsmapFixturePath).readAsStringSync();
    _trie = CharsmapTrie.parse(base64.decode(b64));
  });

  // ── The critical regression: longest- vs shortest-leaf match ─────────────

  group('CharsmapTrie - longest-leaf-match regression (NFD "Việt")', () {
    test('NFD e + combining dot-below + combining circumflex composes to '
        'the fully-composed ệ (U+1EC7), not the partially-composed ẹ', () {
      // U+0065 (e) + U+0323 (combining dot below) + U+0302 (combining
      // circumflex accent) — one extended grapheme cluster with two leaves
      // along its byte path. Taking the shortest match (a literal reading
      // of spm_precompiled's Rust source) would yield 'ẹ' (U+1EB9,
      // dropping the circumflex); the real HuggingFace AutoTokenizer
      // oracle requires the longest match, 'ệ' (U+1EC7).
      const nfdChar = 'ệ';
      final result = _trie.normalize(nfdChar);
      expect(result, equals('ệ'));
    });

    test('full NFD "Việt" word composes every combining-mark cluster', () {
      // "Việt" -- NFD Vietnamese "Việt" as it appears in the
      // plan's smoke corpus (edge_nfd entry).
      const word = 'Việt';
      final result = _trie.normalize(word);
      expect(result, equals('Việt'));
    });
  });

  // ── Fullwidth -> ASCII folding ─────────────────────────────────────────────

  group('CharsmapTrie - fullwidth to ASCII folding', () {
    test('a single fullwidth Latin letter folds to its ASCII equivalent', () {
      // U+FF28 FULLWIDTH LATIN CAPITAL LETTER H -> 'H'.
      expect(_trie.normalize('Ｈ'), equals('H'));
    });

    test('a fullwidth word folds character-by-character', () {
      // "Ｈｅｌｌｏ" (fullwidth) -> "Hello".
      const fullwidthHello = 'Ｈｅｌｌｏ';
      expect(_trie.normalize(fullwidthHello), equals('Hello'));
    });
  });

  // ── Ellipsis normalization ─────────────────────────────────────────────────

  group('CharsmapTrie - ellipsis normalization', () {
    test('U+2026 HORIZONTAL ELLIPSIS expands to three ASCII periods', () {
      expect(_trie.normalize('…'), equals('...'));
    });

    test('ellipsis normalization applies within a larger sentence', () {
      const withEllipsis = 'wait… really?';
      expect(_trie.normalize(withEllipsis), equals('wait... really?'));
    });
  });

  // ── Unmatched / pass-through characters ────────────────────────────────────

  group('CharsmapTrie - unmatched characters pass through unchanged', () {
    test('plain ASCII text is returned unchanged', () {
      expect(_trie.normalize('hello world'), equals('hello world'));
    });

    test('a curly quote with no charsmap entry passes through unchanged', () {
      // U+201C LEFT DOUBLE QUOTATION MARK has no entry in this charsmap
      // (verified empirically against the real fixture) -- it must survive
      // normalize() unchanged rather than being dropped or mis-substituted.
      expect(_trie.normalize('“Hello”'), equals('“Hello”'));
    });
  });

  // ── Empty input ────────────────────────────────────────────────────────────

  group('CharsmapTrie - empty input', () {
    test('empty string normalizes to empty string', () {
      expect(_trie.normalize(''), equals(''));
    });
  });

  // ── transform() ────────────────────────────────────────────────────────────

  group('CharsmapTrie - transform', () {
    test('returns null for a chunk with no trie entry', () {
      expect(_trie.transform('z'), isNull);
    });

    test('returns the replacement text for a chunk with a trie entry', () {
      expect(_trie.transform('Ｈ'), equals('H'));
    });
  });

  // ── Malformed / truncated bytes error handling ────────────────────────────

  group('CharsmapTrie.parse - malformed input', () {
    test('throws FormatException when the blob is shorter than 4 bytes', () {
      expect(
        () => CharsmapTrie.parse(Uint8List.fromList([1, 2, 3])),
        throwsFormatException,
      );
    });

    test('throws FormatException when the declared trie length is not a '
        'multiple of 4', () {
      // Length header declares 1 byte of trie data, which is not a
      // multiple of 4 -- must be rejected rather than silently
      // misinterpreted.
      final blob = Uint8List.fromList([1, 0, 0, 0, 0]);
      expect(() => CharsmapTrie.parse(blob), throwsFormatException);
    });

    test('throws FormatException when the declared trie length overruns the '
        'blob', () {
      // Length header declares 64 bytes of trie data, but only 4 bytes
      // (the header itself) are actually present -- truncated input.
      final blob = Uint8List.fromList([64, 0, 0, 0]);
      expect(() => CharsmapTrie.parse(blob), throwsFormatException);
    });

    test('a minimal single-unit trie with no matching entries normalizes as '
        'identity', () {
      // Header declares 4 bytes of trie data (one root unit, value 0) and
      // an empty replacement-string table. A root unit of 0 has no leaf
      // and offset 0, so any real input byte immediately walks off the
      // single-unit array -- this exercises CharsmapTrie's defensive
      // out-of-range guard in _commonPrefixSearch (a corrupt/degenerate
      // trie must degrade to "no match", not throw RangeError) and
      // confirms normalize() falls back to unchanged pass-through text.
      final blob = Uint8List.fromList([4, 0, 0, 0, 0, 0, 0, 0]);
      final trie = CharsmapTrie.parse(blob);
      expect(trie.normalize('abc'), equals('abc'));
    });
  });
}
