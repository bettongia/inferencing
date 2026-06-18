// Copyright 2026 The Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ignore_for_file: avoid_print

import 'dart:io';

import 'package:betto_inferencing/betto_inferencing.dart';

/// Example demonstrating [BertTokenizer] — the BERT WordPiece tokeniser used
/// internally by [OnnxEmbeddingModel].
///
/// [BertTokenizer] converts raw text into three parallel int64 arrays
/// ([TokenizerOutput.inputIds], [TokenizerOutput.attentionMask],
/// [TokenizerOutput.tokenTypeIds]) ready for ONNX Runtime inference.
///
/// This example downloads the BGE Small En v1.5 vocabulary file via
/// [ModelDownloader] (shared with [OnnxEmbeddingModel]) and then demonstrates:
///   - Encoding a sentence into BERT token IDs
///   - Inspecting the [TokenizerOutput] arrays
///   - Decoding token IDs back to vocabulary strings
///   - Truncation behaviour for inputs that exceed the 510-token budget
///
/// Set BETTO_CACHE to a persistent directory to avoid re-downloading the
/// vocabulary on each run (it is shared with other examples):
///   BETTO_CACHE=$HOME/.cache/betto_examples dart run example/tokenizer.dart
Future<void> main() async {
  // A persistent cache means the vocabulary is downloaded only once and
  // reused across all examples. Fall back to a temporary directory when
  // BETTO_CACHE is not set.
  final cacheDir =
      Platform.environment['BETTO_CACHE'] ??
      Directory.systemTemp.createTempSync('betto_vocab_cache').path;

  print('Cache: $cacheDir');
  print('Ensuring vocabulary is present…');

  // ModelDownloader fetches both the ONNX binary and vocab.txt into cacheDir.
  // Once cached the files are verified by SHA-256 and reused without a
  // network round-trip.
  final downloader = ModelDownloader(allowlist: ModelCatalog());
  final spec = ModelCatalog.lookup(ModelCatalog.defaultModelId);
  final resolved = await downloader.ensure(
    spec,
    cacheDir: cacheDir,
    onProgress: (received, total) {
      final pct = total > 0 ? (received * 100 ~/ total) : 0;
      stdout.write('\r  $pct% ($received / $total bytes)   ');
    },
  );
  stdout.writeln();

  final vocabPath = resolved.filePaths['vocab']!;
  print('Vocabulary: $vocabPath\n');

  final tokenizer = await BertTokenizer.load(vocabPath);

  // ── 1. Basic encoding ──────────────────────────────────────────────────────
  // encode() prepends [CLS] (101), appends [SEP] (102), splits each word into
  // WordPiece sub-tokens, and pads to maxLength (512) with [PAD] (0).
  const sentence = 'Dense retrieval with sentence embeddings.';
  print('Input: "$sentence"');

  final output = tokenizer.encode(sentence);

  // Attention mask is 1 for real tokens, 0 for padding. The first 0 marks the
  // boundary between real tokens and padding.
  final firstPad = output.attentionMask.indexOf(0);
  final realTokenCount = firstPad == -1 ? output.inputIds.length : firstPad;

  print('  Sequence length : ${output.inputIds.length} (max)');
  print('  Real tokens     : $realTokenCount');
  print('  Truncated       : ${output.truncated}');
  print(
    '  inputIds[0..$realTokenCount]'
    ': ${output.inputIds.sublist(0, realTokenCount).toList()}',
  );
  print('');

  // ── 2. Decode token IDs back to vocabulary strings ─────────────────────────
  // decode() maps each integer ID back to its vocabulary entry. Useful for
  // inspecting how the tokeniser splits words into WordPiece sub-tokens
  // (sub-tokens after the first are prefixed with '##').
  final realIds = output.inputIds.sublist(0, realTokenCount).toList();
  final tokens = tokenizer.decode(realIds);
  print('Decoded tokens: $tokens');
  print('');

  // ── 3. WordPiece sub-token splitting ──────────────────────────────────────
  // Words absent from the vocabulary are split into the longest matching
  // sub-token pieces. Rare or technical terms often fragment heavily.
  const technical = 'cosine similarity quantisation embeddings';
  print('Input: "$technical"');
  final techOut = tokenizer.encode(technical);
  final techPad = techOut.attentionMask.indexOf(0);
  final techCount = techPad == -1 ? techOut.inputIds.length : techPad;
  print(
    '  Decoded: ${tokenizer.decode(techOut.inputIds.sublist(0, techCount).toList())}',
  );
  print('');

  // ── 4. Special token IDs ───────────────────────────────────────────────────
  print('Special token IDs:');
  print('  [CLS] = ${BertTokenizer.clsId}  — always the first token');
  print('  [SEP] = ${BertTokenizer.sepId}  — always the last real token');
  print('  [PAD] = ${BertTokenizer.padId}  — fills positions past [SEP]');
  print('  [UNK] = ${BertTokenizer.unkId}  — replaces undecomposable pieces');
  print('');

  // ── 5. Truncation ─────────────────────────────────────────────────────────
  // encode() silently truncates input that would produce more than 510 usable
  // tokens (leaving room for [CLS] and [SEP]). Check truncated to detect this.
  final longText = List.filled(600, 'embedding').join(' ');
  final longOut = tokenizer.encode(longText);
  print('Long input (600 words × "embedding"):');
  print('  Truncated       : ${longOut.truncated}');
  print('  Sequence length : ${longOut.inputIds.length}');
}
