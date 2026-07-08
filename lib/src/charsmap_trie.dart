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

// Traversal structure ported from `spm_precompiled`
// (https://github.com/huggingface/spm_precompiled, Apache-2.0):
// `DoubleArray::common_prefix_search`, `Precompiled::transform`, and
// `Precompiled::normalize_string`'s grapheme-then-char fallback strategy. See
// the repo-root `NOTICE` file for the full attribution.

import 'dart:convert';
import 'dart:typed_data';

import 'package:characters/characters.dart';

/// Parses and applies a SentencePiece/HuggingFace `precompiled_charsmap`
/// normalizer: a "Darts" double-array trie (as used by the `darts-clone`
/// C++ library, https://github.com/s-yata/darts-clone) mapping UTF-8 byte
/// sequences to replacement strings.
///
/// `precompiled_charsmap` is the binary payload of HuggingFace
/// `tokenizer.json`'s `Precompiled` normalizer entry
/// (`normalizer.normalizers[].precompiled_charsmap`, base64-encoded). It
/// implements the NFKC-like text substitution SentencePiece bakes into its
/// trained models — e.g. fullwidth-to-ASCII folding and ellipsis/ligature
/// expansion — as a static lookup table rather than a general Unicode
/// normalization algorithm. It is **not** plain NFKC.
///
/// ## Binary layout
///
/// Confirmed against `google/sentencepiece`'s `normalizer.cc`
/// (`DecodePrecompiledCharsMap`/`EncodePrecompiledCharsMap`) and
/// independently against the Rust `spm_precompiled` crate's `parse()`:
///
/// ```text
/// [0..4)   little-endian uint32: byte length of the trie blob that follows
/// [4..4+n) the trie blob itself: `n` bytes, always a multiple of 4,
///          interpreted as `n/4` little-endian uint32 "units" of a
///          darts-clone double-array trie
/// [4+n..)  the "normalized" string table: a single buffer of UTF-8 bytes
///          holding every replacement string back-to-back, each terminated
///          by a NUL (0x00) byte. A trie leaf's value is a byte offset into
///          this table; the replacement text is read from that offset up to
///          (excluding) the next NUL byte.
/// ```
///
/// Each 32-bit darts-clone unit (`include/darts.h`,
/// `Details::DoubleArrayUnit`) packs:
///
/// - `hasLeaf`: bit 8 (`(unit >> 8) & 1`)
/// - `value`: low 31 bits (`unit & ((1 << 31) - 1)`) — valid only when
///   reading the *leaf* unit reached via `offset` below
/// - `label`: MSB + low byte (`unit & ((1 << 31) | 0xFF)`) — the transition
///   byte (or, with the MSB set, "no valid label", which is how leaf units
///   make `label` never collide with a real 0..255 byte value)
/// - `offset`: `(unit >> 10) << ((unit & (1 << 9)) >> 6)` — an XOR-linked
///   offset to the unit's child units (the "double array" scheme; see
///   darts-clone's README)
///
/// ## Traversal algorithm
///
/// SentencePiece's own C++ (`Normalizer::NormalizePrefix`) does a flat
/// byte-position scan over the trie, picking the longest match. That is
/// **not** what actually produces byte-exact HuggingFace `AutoTokenizer`
/// output: `tokenizer.json`-based tokenizers (via the `tokenizers` Rust
/// crate) go through `spm_precompiled`'s `Precompiled::normalize_string`
/// instead, which:
///
/// 1. Splits the input into Unicode extended grapheme clusters (UAX #29),
///    not a flat byte stream.
/// 2. For each cluster whose UTF-8 encoding is under 6 bytes, looks up the
///    *whole cluster* in the trie first (see [transform]).
/// 3. On no match (or a cluster ≥ 6 bytes), falls back to looking up each
///    individual Unicode scalar value (char) within the cluster on its own;
///    unmatched characters pass through unchanged.
///
/// [normalize] implements this grapheme-then-char algorithm — matching the
/// real oracle, not a literal reading of SentencePiece's C++ source.
class CharsmapTrie {
  CharsmapTrie._(this._units, this._normalized);

  /// Parses a raw (already base64-decoded) `precompiled_charsmap` blob.
  ///
  /// Throws [FormatException] if [blob] is too short to contain a valid
  /// length header, or if the declared trie length is not a multiple of 4
  /// bytes or overruns the blob — both are unambiguous signs of truncated or
  /// corrupt input, and are rejected here rather than left to fail
  /// unpredictably during a later [normalize] call.
  factory CharsmapTrie.parse(Uint8List blob) {
    if (blob.length < 4) {
      throw const FormatException(
        'precompiled_charsmap blob too short: must be at least 4 bytes '
        '(the little-endian trie-length header).',
      );
    }
    final data = ByteData.sublistView(blob);
    final trieByteLength = data.getUint32(0, Endian.little);
    if (trieByteLength % 4 != 0 || 4 + trieByteLength > blob.length) {
      throw FormatException(
        'precompiled_charsmap trie length $trieByteLength is not a '
        'multiple of 4, or overruns the blob (blob length ${blob.length}). '
        'This indicates truncated or corrupt charsmap bytes.',
      );
    }
    final unitCount = trieByteLength ~/ 4;
    final units = Uint32List(unitCount);
    for (var i = 0; i < unitCount; i++) {
      units[i] = data.getUint32(4 + i * 4, Endian.little);
    }
    final normalized = blob.sublist(4 + trieByteLength);
    return CharsmapTrie._(units, normalized);
  }

  /// The darts-clone double-array "units" — one 32-bit int per trie node.
  final Uint32List _units;

  /// The NUL-delimited table of UTF-8 replacement strings that trie leaf
  /// values are byte offsets into.
  final Uint8List _normalized;

  static bool _hasLeaf(int unit) => ((unit >> 8) & 1) == 1;
  static int _value(int unit) => unit & ((1 << 31) - 1);
  static int _label(int unit) => unit & ((1 << 31) | 0xFF);
  static int _offset(int unit) => (unit >> 10) << ((unit & (1 << 9)) >> 6);

  /// Walks the trie over every byte of [key], returning the leaf `value()`
  /// of *every* node along the walk that has a leaf child (not just the
  /// final one) — mirroring `spm_precompiled`'s
  /// `DoubleArray::common_prefix_search`, including its choice to keep
  /// walking past the first leaf rather than stopping there.
  ///
  /// Defensive against corrupt trie data: an out-of-range node index (which
  /// a hand-crafted or bit-flipped trie could produce even though
  /// [CharsmapTrie.parse]'s own header checks passed) ends the walk and
  /// returns whatever leaves were already found, rather than throwing
  /// [RangeError] — malformed *lookup* data degrades to "no further match"
  /// instead of crashing the caller.
  List<int> _commonPrefixSearch(List<int> key) {
    var nodePos = 0;
    final results = <int>[];
    var unit = _units[nodePos];
    nodePos ^= _offset(unit);
    for (final c in key) {
      if (c == 0) break;
      final candidate = nodePos ^ c;
      if (candidate < 0 || candidate >= _units.length) return results;
      nodePos = candidate;
      unit = _units[nodePos];
      if (_label(unit) != c) return results;
      final childPos = nodePos ^ _offset(unit);
      if (childPos < 0 || childPos >= _units.length) return results;
      nodePos = childPos;
      if (_hasLeaf(unit)) {
        results.add(_value(_units[nodePos]));
      }
    }
    return results;
  }

  /// Looks up [chunk] (a single grapheme cluster or a single character) as a
  /// whole key. Returns the replacement text if the trie has an entry for
  /// it, or `null` if there is no match.
  ///
  /// **Longest-, not shortest-, leaf match is required — do not "fix" this
  /// back based on a literal reading of `spm_precompiled`'s source.** A
  /// literal reading of `spm_precompiled::Precompiled::transform`'s Rust
  /// source takes `results[0]` (the *shortest* matched prefix — leaves are
  /// appended to the results vector in the order the byte-walk encounters
  /// them, which is shortest-to-longest). That is empirically wrong: NFD
  /// `"Việt"`'s `e` + combining dot-below (U+0323) + combining circumflex
  /// (U+0302) — one extended grapheme cluster — has *two* leaves along its
  /// byte path: a shorter one for `e`+dot-below alone (mapping to the
  /// partially-composed `ẹ`) and a longer one for the full three-codepoint
  /// sequence (mapping to the fully composed `ệ`). Taking the shortest
  /// match yields `ẹ`, silently dropping the circumflex, and diverges from
  /// real HuggingFace `AutoTokenizer` output. Taking the **longest** match
  /// (`results.last`, matching SentencePiece C++'s own `NormalizePrefix`
  /// longest-match rule in `normalizer.cc`) yields `ệ` and reproduces the
  /// real oracle exactly. See `test/charsmap_trie_test.dart` for the
  /// regression test this gotcha is pinned down by.
  String? transform(String chunk) {
    final keyBytes = utf8.encode(chunk);
    final results = _commonPrefixSearch(keyBytes);
    if (results.isEmpty) return null;
    final start = results.last;
    var end = start;
    while (end < _normalized.length && _normalized[end] != 0) {
      end++;
    }
    return utf8.decode(_normalized.sublist(start, end));
  }

  /// Applies the charsmap substitution to [original].
  ///
  /// Follows `spm_precompiled::Precompiled::normalize_string`'s
  /// grapheme-cluster-first, falling-back-to-single-char algorithm: for each
  /// extended grapheme cluster, try a whole-cluster replacement first (only
  /// attempted when the cluster's UTF-8 encoding is under 6 bytes, matching
  /// the reference implementation's own threshold), then fall back to
  /// looking up each character in the cluster individually, keeping
  /// unmatched characters unchanged.
  ///
  /// Deliberately does **not** perform SentencePiece's
  /// whitespace-collapsing, dummy-prefix, or Metaspace-escaping steps —
  /// those are the caller's responsibility (see [XlmRobertaTokenizer] in
  /// `xlmr_tokenizer.dart`, which applies them immediately after this
  /// substitution and before handing the result to
  /// `dart_sentencepiece_tokenizer`'s own tokenizer).
  String normalize(String original) {
    final buffer = StringBuffer();
    for (final grapheme in original.characters) {
      final graphemeUtf8Length = utf8.encode(grapheme).length;
      if (graphemeUtf8Length < 6) {
        final whole = transform(grapheme);
        if (whole != null) {
          buffer.write(whole);
          continue;
        }
      }
      // Fall back to per-character (per Unicode scalar value) lookup within
      // the grapheme cluster.
      for (final rune in grapheme.runes) {
        final char = String.fromCharCode(rune);
        final replacement = transform(char);
        buffer.write(replacement ?? char);
      }
    }
    return buffer.toString();
  }
}
