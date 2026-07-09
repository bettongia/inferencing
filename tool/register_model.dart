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
// One-off dev tool (not run by CI) that downloads a `ModelCatalog` entry's
// asset files (`model.onnx`, tokenizer asset) directly from the exact
// `resolve/main/...` HuggingFace URL `ModelSpec` will use at runtime, and
// computes their SHA-256 digests using the same `package:crypto` algorithm
// `ModelDownloader` uses for checksum verification -- so the hash this tool
// prints is *guaranteed* to match what a real download-on-demand run would
// verify, not an approximation from a different source (e.g. a HuggingFace
// `X-Linked-ETag`/xet-hash header, which uses a different content-addressing
// scheme and is not interchangeable with a plain SHA-256 of the file bytes).
//
// This exists for the WI-4 "hand-off point" (see
// `docs/plans/plan_0_06_wi4_multilingual_embedding_model.md` in the `kmdb`
// repo, Phase 2): registering a new model in `ModelCatalog` requires real
// checksums, but this project's implementer sandbox may not always have
// network access to `huggingface.co`. Run this script standalone (as the
// hand-off asks) to produce the values that belong in `model_catalog.dart`'s
// `_multilingualE5Small*Sha256` constants:
//
// ```bash
// dart run tool/register_model.dart
// ```
//
// Downloaded files are cached under `tool/.cache/register_model/` --
// re-running the script reuses the cached bytes rather than re-downloading
// (mirrors `tool/generate_xlmr_parity_corpus.dart`'s `_fetchArchive` caching
// pattern). Delete that directory to force a fresh download (e.g. after an
// upstream model update).
import 'dart:io';

import 'package:convert/convert.dart' show AccumulatorSink;
import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;

const _cacheDir = 'tool/.cache/register_model';

/// The asset files that make up `multilingual-e5-small`'s `ModelCatalog`
/// entry -- same two files (`onnx`, `vocab`/tokenizer asset) `ModelSpec.files`
/// declares in `lib/src/model_catalog.dart`, at the identical URLs.
///
/// Kept as a plain, hardcoded list (rather than importing `ModelCatalog`
/// itself) so this tool can be run *before* the catalog entry has real
/// checksums -- it produces the values that then get pasted into
/// `model_catalog.dart`, not the other way around.
const _files = <(String name, String url)>[
  (
    'model.onnx',
    'https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx',
  ),
  (
    'tokenizer.json',
    'https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/tokenizer.json',
  ),
];

Future<void> main() async {
  Directory(_cacheDir).createSync(recursive: true);

  // ignore: avoid_print
  print(
    'Registering multilingual-e5-small -- downloading and verifying '
    '${_files.length} file(s).\n',
  );

  final results = <String, String>{};
  for (final (name, url) in _files) {
    final digest = await _downloadAndHash(name, url);
    results[name] = digest;
  }

  // ignore: avoid_print
  print('\n=== SHA-256 digests (paste into model_catalog.dart) ===');
  for (final entry in results.entries) {
    // ignore: avoid_print
    print('${entry.key}: ${entry.value}');
  }
}

/// Downloads (or reuses a cached copy of) the file at [url], streaming it to
/// `tool/.cache/register_model/$fileName` and computing its SHA-256 digest
/// incrementally (rather than buffering the whole file in memory -- `
/// model.onnx` is ~470 MB).
///
/// Returns the lowercase hex SHA-256 digest -- byte-for-byte the same
/// algorithm and library (`package:crypto`) `ModelDownloader._isValid` uses,
/// so a value printed here is guaranteed to verify against a real download at
/// runtime.
Future<String> _downloadAndHash(String fileName, String url) async {
  final cacheFile = File('$_cacheDir/$fileName');

  if (cacheFile.existsSync()) {
    // ignore: avoid_print
    print('Using cached file: ${cacheFile.path}');
  } else {
    // ignore: avoid_print
    print('Downloading: $url');
    final request = http.Request('GET', Uri.parse(url));
    final client = http.Client();
    try {
      final response = await client.send(request);
      if (response.statusCode != 200) {
        throw StateError(
          'Failed to download $url: HTTP ${response.statusCode}',
        );
      }
      // Stream to the temp file directly rather than buffering all bytes in
      // memory -- model.onnx is large enough (~470 MB) that this matters.
      final tempFile = File('${cacheFile.path}.part');
      final sink = tempFile.openWrite();
      var received = 0;
      final total = response.contentLength ?? -1;
      await for (final chunk in response.stream) {
        sink.add(chunk);
        received += chunk.length;
        if (total > 0) {
          final pct = (received / total * 100).toStringAsFixed(1);
          stdout.write('\r  $fileName: $received / $total bytes ($pct%)      ');
        } else {
          stdout.write('\r  $fileName: $received bytes      ');
        }
      }
      await sink.flush();
      await sink.close();
      stdout.writeln();
      // Rename only after the full download completes -- avoids leaving a
      // truncated file at the final path if the process is interrupted
      // mid-download, matching ModelDownloader's own temp-file-then-rename
      // crash-safety pattern.
      await tempFile.rename(cacheFile.path);
    } finally {
      client.close();
    }
  }

  // Compute the SHA-256 digest incrementally via a byte-stream sink rather
  // than reading the whole file into memory at once.
  final digestSink = AccumulatorSink<Digest>();
  final input = sha256.startChunkedConversion(digestSink);
  await for (final chunk in cacheFile.openRead()) {
    input.add(chunk);
  }
  input.close();
  final digest = digestSink.events.single;
  return digest.toString();
}
