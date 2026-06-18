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

@TestOn('vm')
library;

import 'dart:io';

import 'package:betto_inferencing/betto_inferencing.dart';
import 'package:betto_lexical/betto_lexical.dart' show Tokenizer;
import 'package:test/test.dart';

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Path to the synthetic fixture vocab used for unit tests.
///
/// Token index map (0-based line index = token ID):
///   0=[PAD], 1-99=[unused1..99], 100=[UNK], 101=[CLS], 102=[SEP],
///   103=hello, 104=world, 105=je, 106=##ky, 107=##ll, 108=test,
///   109=cafe, 110=hi
String get _vocabPath {
  // Directory.current is the package root when dart test is invoked.
  return '${Directory.current.path}/test/fixtures/vocab.txt';
}

void main() {
  late BertTokenizer tokenizer;

  setUpAll(() async {
    tokenizer = await BertTokenizer.load(_vocabPath);
  });

  // ── Sentinels ──────────────────────────────────────────────────────────────

  group('BertTokenizer - sentinels', () {
    test('encode() starts with [CLS] token (id=101)', () {
      final out = tokenizer.encode('hello world');
      expect(out.inputIds[0], equals(BertTokenizer.clsId));
    });

    test('encode() ends with [SEP] token at the last real position', () {
      final out = tokenizer.encode('hello world');
      final lastRealIdx = out.attentionMask.lastIndexWhere((m) => m == 1);
      expect(out.inputIds[lastRealIdx], equals(BertTokenizer.sepId));
    });

    test('output has exactly maxLength (512) elements', () {
      final out = tokenizer.encode('hello world');
      expect(out.inputIds.length, equals(512));
      expect(out.attentionMask.length, equals(512));
      expect(out.tokenTypeIds.length, equals(512));
    });

    test('token_type_ids are all zeros (single-segment input)', () {
      final out = tokenizer.encode('any text here');
      expect(out.tokenTypeIds.every((v) => v == 0), isTrue);
    });
  });

  // ── Known token IDs ────────────────────────────────────────────────────────

  group('BertTokenizer - known token IDs (fixture vocab)', () {
    test('hello (103) and world (104) have correct fixture vocabulary IDs', () {
      final out = tokenizer.encode('hello world');
      // inputIds[0]=CLS, [1]=hello, [2]=world, [3]=SEP
      expect(out.inputIds[1], equals(103)); // 'hello'
      expect(out.inputIds[2], equals(104)); // 'world'
    });

    test('jekyll WordPiece splits to [je(105), ##ky(106), ##ll(107)]', () {
      final out = tokenizer.encode('jekyll');
      // CLS, je(105), ##ky(106), ##ll(107), SEP, PAD...
      expect(out.inputIds[1], equals(105));
      expect(out.inputIds[2], equals(106));
      expect(out.inputIds[3], equals(107));
      expect(out.inputIds[4], equals(BertTokenizer.sepId));
    });
  });

  // ── Unknown words ──────────────────────────────────────────────────────────

  group('BertTokenizer - unknown words', () {
    test('word absent from vocabulary and non-decomposable maps to [UNK]', () {
      // 'xyz' has no entry in the fixture vocab and no valid WordPiece splits.
      final out = tokenizer.encode('xyz');
      // CLS, UNK(100), SEP
      expect(out.inputIds[1], equals(BertTokenizer.unkId));
      expect(out.inputIds[2], equals(BertTokenizer.sepId));
    });
  });

  // ── Normalisation ──────────────────────────────────────────────────────────

  group('BertTokenizer - normalisation', () {
    test('input text is lowercased before tokenisation', () {
      final lower = tokenizer.encode('hello');
      final upper = tokenizer.encode('HELLO');
      // Both should produce the same token IDs.
      expect(upper.inputIds[1], equals(lower.inputIds[1]));
    });

    test('combining accent characters (U+0300-U+036F) are stripped', () {
      // U+0301 (COMBINING ACUTE ACCENT) lies in the 0x0300-0x036F strip range.
      // 'café' (e + combining acute in NFD) normalises to 'cafe' (ID 109).
      final out = tokenizer.encode('café');
      expect(out.inputIds[1], equals(109)); // 'cafe' after stripping
      expect(out.inputIds[2], equals(BertTokenizer.sepId));
    });
  });

  // ── Truncation ────────────────────────────────────────────────────────────

  group('BertTokenizer - truncation', () {
    test('long text exceeding 510 usable tokens is truncated', () {
      // 600 single-token words exceeds the 510 usable-token budget.
      final longText = List.filled(600, 'hello').join(' ');
      final out = tokenizer.encode(longText);
      expect(out.truncated, isTrue);
      final lastRealIdx = out.attentionMask.lastIndexWhere((m) => m == 1);
      expect(out.inputIds[lastRealIdx], equals(BertTokenizer.sepId));
      expect(out.inputIds.length, equals(512));
    });

    test('text fitting exactly in 510 usable tokens is not truncated', () {
      // 510 single-token words fills the usable budget exactly.
      final text = List.filled(510, 'hello').join(' ');
      final out = tokenizer.encode(text);
      expect(out.truncated, isFalse);
      expect(out.attentionMask.every((m) => m == 1), isTrue);
    });
  });

  // ── Empty input ────────────────────────────────────────────────────────────

  group('BertTokenizer - empty input', () {
    test('empty string produces [CLS][SEP] only, no error', () {
      final out = tokenizer.encode('');
      expect(out.inputIds[0], equals(BertTokenizer.clsId));
      expect(out.inputIds[1], equals(BertTokenizer.sepId));
      for (var i = 2; i < 512; i++) {
        expect(out.inputIds[i], equals(BertTokenizer.padId));
        expect(out.attentionMask[i], equals(0));
      }
      expect(out.truncated, isFalse);
    });

    test('whitespace-only string produces [CLS][SEP] only', () {
      final out = tokenizer.encode('   \t\n   ');
      expect(out.inputIds[0], equals(BertTokenizer.clsId));
      expect(out.inputIds[1], equals(BertTokenizer.sepId));
      expect(out.truncated, isFalse);
    });
  });

  // ── Empty words from tokenizer ─────────────────────────────────────────────

  group('BertTokenizer - empty words from tokenizer', () {
    test('empty strings in tokenizer output are silently skipped', () async {
      // A tokenizer that emits empty strings around a real word. BertTokenizer
      // must skip them without panicking or producing spurious tokens.
      final tok = await BertTokenizer.load(
        _vocabPath,
        tokenizer: const _EmptyWordTokenizer(),
      );
      final out = tok.encode('hello');
      // Empty strings are skipped; only 'hello' (103) should appear.
      expect(out.inputIds[0], equals(BertTokenizer.clsId));
      expect(out.inputIds[1], equals(103));
      expect(out.inputIds[2], equals(BertTokenizer.sepId));
      expect(out.truncated, isFalse);
    });
  });

  // ── Attention mask ─────────────────────────────────────────────────────────

  group('BertTokenizer - attention mask', () {
    test('attention mask is 1 for real tokens and 0 for padding', () {
      final out = tokenizer.encode('hello world');
      // CLS(1), hello(1), world(1), SEP(1), then all zeros.
      expect(out.attentionMask[0], equals(1)); // CLS
      expect(out.attentionMask[1], equals(1)); // hello
      expect(out.attentionMask[2], equals(1)); // world
      expect(out.attentionMask[3], equals(1)); // SEP
      for (var i = 4; i < 512; i++) {
        expect(out.attentionMask[i], equals(0));
      }
    });
  });

  // ── Decode ────────────────────────────────────────────────────────────────

  group('BertTokenizer - decode', () {
    test('decode returns vocabulary strings for known IDs', () {
      final decoded = tokenizer.decode([
        BertTokenizer.clsId,
        103,
        104,
        BertTokenizer.sepId,
      ]);
      expect(decoded, equals(['[CLS]', 'hello', 'world', '[SEP]']));
    });

    test('decode maps IDs absent from the vocabulary to [UNK] string', () {
      final decoded = tokenizer.decode([99999]);
      expect(decoded, equals(['[UNK]']));
    });
  });

  // ── maxLength override ─────────────────────────────────────────────────────

  group('BertTokenizer - maxLength override', () {
    test('custom maxLength truncates output to the specified length', () async {
      final shortTok = await BertTokenizer.load(_vocabPath, maxLength: 10);
      final out = shortTok.encode('hello world');
      expect(out.inputIds.length, equals(10));
      expect(out.attentionMask.length, equals(10));
      expect(out.tokenTypeIds.length, equals(10));
    });
  });

  // ── Tokenizer substitution ─────────────────────────────────────────────────

  group('BertTokenizer - tokenizer substitution', () {
    test(
      'custom Tokenizer is used for word segmentation without error',
      () async {
        final customTokenizer = await BertTokenizer.load(
          _vocabPath,
          tokenizer: const _WhitespaceTokenizer(),
        );
        final out = customTokenizer.encode('hello world');
        expect(out.inputIds[0], equals(BertTokenizer.clsId));
        expect(out.truncated, isFalse);
      },
    );
  });
}

// ── Test Tokenizers ───────────────────────────────────────────────────────────

/// Splits on whitespace only — verifies the Tokenizer injection point.
final class _WhitespaceTokenizer implements Tokenizer {
  const _WhitespaceTokenizer();

  @override
  List<String> tokenise(String text) =>
      text.split(RegExp(r'\s+')).where((s) => s.isNotEmpty).toList();
}

/// Emits empty strings around a single token — exercises the
/// `if (word.isEmpty) continue` guard inside [BertTokenizer.encode].
final class _EmptyWordTokenizer implements Tokenizer {
  const _EmptyWordTokenizer();

  @override
  List<String> tokenise(String text) => ['', text.trim(), ''];
}
