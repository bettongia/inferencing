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

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';
import 'package:meta/meta.dart';

import 'bert_tokenizer.dart' show TokenizerOutput;
import 'charsmap_trie.dart';

/// XLM-RoBERTa-family SentencePiece/Unigram tokenizer, e.g. for
/// `multilingual-e5-small`.
///
/// Composes a from-scratch [CharsmapTrie] normalizer (the one piece of
/// SentencePiece's pipeline this project could not find a working Dart
/// implementation of anywhere — see `CharsmapTrie`'s own doc comment) with
/// `dart_sentencepiece_tokenizer`'s public API for everything else: vocab
/// loading, Unigram Viterbi decoding, and BOS/EOS post-processing.
///
/// ## Why this class exists instead of using `dart_sentencepiece_tokenizer`
/// directly
///
/// `dart_sentencepiece_tokenizer` parses HuggingFace `tokenizer.json`'s
/// `Precompiled` normalizer type (the charsmap trie bytes) but never applies
/// it — `HuggingFaceTokenizerLoader`'s JSON-loading path silently drops the
/// charsmap, making its own `SpNormalizer` an unconditional
/// (charsmap-less) pass-through for models like `multilingual-e5-small`.
/// See this package's `README.md` ("Why not `dart_sentencepiece_tokenizer`
/// alone") for the full write-up of this and a second, independent defect
/// this class also works around.
///
/// [SentencePieceTokenizer]'s real constructor is private and every public
/// factory routes through internal, non-injectable construction — there is
/// no way to subclass or swap in a corrected normalizer. Composition (running
/// our own charsmap normalization *before* handing text to the library's
/// public `encode()`) is therefore the only viable integration seam, and is
/// what [encode] does.
///
/// ## Pipeline
///
/// [encode] applies, in order:
///
/// 1. [CharsmapTrie.normalize] — the charsmap substitution described above.
/// 2. Whitespace-run collapse (two-or-more plain spaces → one).
/// 3. Metaspace escaping (prepend a leading space if absent; replace every
///    space with `▁`, U+2581).
/// 4. `dart_sentencepiece_tokenizer`'s own `SentencePieceTokenizer.encode()`.
///
/// **Why steps 2–3 are done manually here rather than trusted to the
/// library's own `SpNormalizer`:** `dart_sentencepiece_tokenizer`'s
/// HuggingFace-JSON metadata parser derives its
/// `addDummyPrefix`/`removeExtraWhitespaces`/`escapeWhitespaces` flags by
/// pattern-matching the `normalizer` section of `tokenizer.json`, but
/// `multilingual-e5-small`'s `tokenizer.json` puts this configuration
/// entirely in `pre_tokenizer` (`{"type": "Metaspace", "replacement": "▁",
/// "add_prefix_space": true}`) instead — a section the parser never reads.
/// All three flags therefore come back `false`, and the library's own
/// whitespace/prefix handling silently no-ops for this file. This is a
/// second, independent defect from the already-known charsmap-drop one
/// (confirmed empirically: feeding the library literal pre-escaped text like
/// `"▁Hello"` produces the correct single-token id, while plain `"Hello"` or
/// `" Hello"` does not) — so this class must own both normalization
/// concerns, not just the charsmap.
///
/// Note: this does **not** implement a shared `ModelTokenizer` interface —
/// that type does not exist yet (it is introduced by a separate, later
/// embedding-model work item that in turn depends on this class existing
/// first).
class XlmRobertaTokenizer {
  XlmRobertaTokenizer._(this._charsmapTrie, this._tokenizer, this._maxLength);

  final CharsmapTrie _charsmapTrie;
  final SentencePieceTokenizer _tokenizer;
  final int _maxLength;

  /// Loads a tokenizer from a HuggingFace `tokenizer.json` file at
  /// [tokenizerJsonPath].
  ///
  /// Extracts the `Precompiled` normalizer's `precompiled_charsmap` bytes
  /// (`normalizer.normalizers[]`, `type: "Precompiled"`) to build a
  /// [CharsmapTrie], and separately passes the full JSON to
  /// `HuggingFaceTokenizerLoader.fromJsonString` to build the underlying
  /// [SentencePieceTokenizer] (vocabulary, Unigram model, BOS/EOS
  /// configuration).
  ///
  /// [maxLength] is the maximum sequence length, including the `<s>`/`</s>`
  /// sentinel tokens. Defaults to 512, matching both [BertTokenizer]'s own
  /// default and `multilingual-e5-small`'s published `max_seq_length` (the
  /// model's `tokenizer.json` itself declares no `truncation`/`padding`
  /// section, so there is no stronger signal to defer to).
  ///
  /// Throws [FormatException] if [tokenizerJsonPath]'s JSON has no
  /// `Precompiled` normalizer entry, or if its `precompiled_charsmap` bytes
  /// are malformed (see [CharsmapTrie.parse]).
  static Future<XlmRobertaTokenizer> load(
    String tokenizerJsonPath, {
    int maxLength = 512,
  }) async {
    // coverage:ignore-start
    // Requires the real ~17 MB multilingual-e5-small tokenizer.json (250k-entry
    // vocab) to exercise meaningfully — covered by the network-gated
    // integration test in integration_test_app/, not make coverage. See this
    // package's README for why that fixture isn't committed.
    final raw = await File(tokenizerJsonPath).readAsString();
    final tokenizerJson = jsonDecode(raw) as Map<String, dynamic>;
    final charsmapTrie = _extractCharsmapTrie(tokenizerJson);
    final tokenizer = HuggingFaceTokenizerLoader.fromJsonString(raw);
    return XlmRobertaTokenizer._(charsmapTrie, tokenizer, maxLength);
    // coverage:ignore-end
  }

  /// Extracts and parses the `Precompiled` normalizer's charsmap bytes from
  /// a parsed `tokenizer.json` map.
  ///
  /// Throws [FormatException] if no `Precompiled` normalizer entry is
  /// present (the `normalizer` section is expected to be a `Sequence` whose
  /// entries include one with `"type": "Precompiled"`, per
  /// `multilingual-e5-small`'s own `tokenizer.json` shape).
  static CharsmapTrie _extractCharsmapTrie(Map<String, dynamic> tokenizerJson) {
    final normalizerSection = tokenizerJson['normalizer'];
    final normalizers = normalizerSection is Map<String, dynamic>
        ? normalizerSection['normalizers']
        : null;
    if (normalizers is! List) {
      throw const FormatException(
        'tokenizer.json has no normalizer.normalizers list — expected a '
        'Sequence normalizer containing a Precompiled entry.',
      );
    }
    final precompiled = normalizers
        .cast<Map<String, dynamic>>()
        .where((n) => n['type'] == 'Precompiled')
        .firstOrNull;
    final charsmapB64 = precompiled?['precompiled_charsmap'];
    if (charsmapB64 is! String) {
      throw const FormatException(
        'tokenizer.json has no normalizer.normalizers[] entry with '
        'type "Precompiled" and a precompiled_charsmap field — this loader '
        'only supports XLM-RoBERTa-family tokenizer.json files that use a '
        'Precompiled charsmap normalizer.',
      );
    }
    return CharsmapTrie.parse(base64.decode(charsmapB64));
  }

  /// Encodes [text] into a [TokenizerOutput] ready for ONNX inference.
  ///
  /// Reuses the same [TokenizerOutput] type [BertTokenizer.encode] returns
  /// (not a parallel type), so both tokenizers already share an identical
  /// concrete output shape.
  ///
  /// - [TokenizerOutput.tokenTypeIds] is all-zeros: XLM-RoBERTa/RoBERTa
  ///   models don't use segment ids, but the field is required by
  ///   [TokenizerOutput] — this is a direct widen of
  ///   `Encoding.typeIds`, which is already all-zeros for single-segment
  ///   input in `dart_sentencepiece_tokenizer`, not invented data.
  /// - Padding uses the loaded vocabulary's own pad id (`1` for
  ///   `multilingual-e5-small`'s `<pad>`), sourced automatically by
  ///   `SentencePieceTokenizer` — not the BERT-specific `padId = 0`
  ///   [BertTokenizer] uses.
  /// - [TokenizerOutput.truncated] is derived via two encode passes (see
  ///   below), because `Encoding` exposes no overflow signal and a padded
  ///   output is always exactly [maxLength] tokens long regardless of
  ///   whether truncation actually occurred.
  ///
  /// All three output arrays have exactly the `maxLength` passed to [load]
  /// elements.
  TokenizerOutput encode(String text) {
    final normalizedText = normalizeForTokenization(_charsmapTrie, text);

    // coverage:ignore-start
    // Requires the real multilingual-e5-small vocabulary to exercise
    // meaningfully — covered by the network-gated integration test in
    // integration_test_app/, not make coverage.
    //
    // `truncated` derivation is a two-pass process. `enablePadding`/
    // `enableTruncation` mutate *persistent* instance state on the shared
    // `SentencePieceTokenizer` (they are not per-call arguments), so an
    // explicit noPadding()/noTruncation() reset is mandatory on *every* call
    // — without it, the "unbounded" first pass below would still be bounded
    // by whatever config a *previous* encode() call left in place, and
    // `truncated` would silently stick `false` after the first call.
    _tokenizer.noPadding();
    _tokenizer.noTruncation();

    // Pass 1 (unbounded): get the true token count with no truncation, to
    // detect whether pass 2's truncation will actually cut anything.
    final rawLength = _tokenizer
        .encode(normalizedText, addSpecialTokens: true)
        .ids
        .length;
    final truncated = rawLength > _maxLength;

    // Pass 2 (bounded): the real, padded/truncated result. Both passes use
    // identical addSpecialTokens: true so rawLength (which includes the
    // <s>/</s> sentinels) is compared on equal terms against the maxLength
    // the bounded pass truncates to.
    _tokenizer
      ..enablePadding(direction: SpPaddingDirection.right, length: _maxLength)
      ..enableTruncation(
        maxLength: _maxLength,
        direction: SpTruncationDirection.right,
      );
    final encoding = _tokenizer.encode(normalizedText, addSpecialTokens: true);

    return TokenizerOutput(
      inputIds: Int64List.fromList(encoding.ids),
      attentionMask: Int64List.fromList(encoding.attentionMask),
      tokenTypeIds: Int64List.fromList(encoding.typeIds),
      truncated: truncated,
    );
    // coverage:ignore-end
  }

  /// Applies the charsmap substitution, whitespace-run collapse, and
  /// Metaspace escaping steps of [encode]'s pipeline (steps 1–3), without
  /// requiring a loaded [SentencePieceTokenizer]/vocabulary.
  ///
  /// Exposed (rather than kept private) purely so these offline,
  /// vocab-independent steps can be unit-tested directly against the small
  /// committed charsmap fixture, without needing the full ~17 MB
  /// `tokenizer.json` — see `test/xlmr_tokenizer_test.dart`. Not intended
  /// for use outside this package or its tests.
  @visibleForTesting
  static String normalizeForTokenization(CharsmapTrie trie, String text) {
    final substituted = trie.normalize(text);
    final collapsed = _collapseSpaceRuns(substituted);
    return _metaspace(collapsed);
  }

  /// Collapses runs of two or more plain space characters (U+0020) down to
  /// one, replicating `multilingual-e5-small`'s `tokenizer.json`
  /// `normalizer.normalizers[1]` stage (a literal `Replace` node for the
  /// pattern `" {2,}"` → `" "`).
  ///
  /// Deliberately narrower than SentencePiece C++'s own generic whitespace
  /// collapsing (which recognises many whitespace code points and also
  /// trims leading/trailing runs) — this mirrors only what this
  /// `tokenizer.json` itself declares.
  static String _collapseSpaceRuns(String text) =>
      text.replaceAll(RegExp(' {2,}'), ' ');

  /// Applies `multilingual-e5-small`'s `tokenizer.json` `pre_tokenizer`
  /// stage: a `Metaspace` pre-tokenizer with `replacement: "▁"` and
  /// `add_prefix_space: true`.
  ///
  /// Prepends a single leading space if [text] doesn't already start with
  /// one, then replaces every space with the metaspace symbol (U+2581).
  ///
  /// **Empty input is a special case, found by WI-4's broader parity
  /// corpus** (`test/fixtures/xlmr_parity_corpus.json`'s `edge_empty`
  /// entry): real `AutoTokenizer` output for `""` is just `[<s>, </s>]`
  /// (`[0, 2]`) with no content token at all, but naively adding the dummy
  /// prefix here would turn `""` into `"▁"`, which the real vocabulary
  /// treats as a *valid standalone piece* (id 6) -- producing a spurious
  /// extra content token (`[0, 6, 2]`) that fails byte-exact parity.
  /// HuggingFace's `tokenizers` Rust `Metaspace` pre-tokenizer only adds the
  /// prefix space when pre-tokenizing an actual word/split; a fully empty
  /// input has no splits to begin with, so the prefix is never added.
  /// Returning `''` unchanged for empty input reproduces that behaviour
  /// without needing to port the Rust library's split-detection machinery.
  static String _metaspace(String text) {
    if (text.isEmpty) return text;
    final withPrefix = text.startsWith(' ') ? text : ' $text';
    return withPrefix.replaceAll(' ', '▁');
  }
}
