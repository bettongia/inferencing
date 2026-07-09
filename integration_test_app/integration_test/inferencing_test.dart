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
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:betto_inferencing/betto_inferencing.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Resolve the model cache directory. CI sets MODEL_CACHE_DIR to a stable
  // path that survives between runs (so actions/cache can restore it). Locally
  // we fall back to $HOME/.cache so the ~127 MB download is also reused across
  // test runs, with the system temp as a last resort.
  final cacheDir =
      Platform.environment['MODEL_CACHE_DIR'] ??
      (Platform.environment['HOME'] != null
          ? '${Platform.environment['HOME']}/.cache/betto_inferencing_models'
          : '${Directory.systemTemp.path}/betto_inferencing_models');
  final modelCache = Directory(cacheDir)..createSync(recursive: true);

  // ── Pure-Dart smoke tests ─────────────────────────────────────────────────
  // These run without network access or native libraries and serve as a fast
  // sanity check that the Dart layer wired up correctly on the target platform.

  group('ModelCatalog', () {
    test('lookup returns the validated BGE Small En spec', () {
      final spec = ModelCatalog.lookup('bge-small-en-v1.5');
      expect(spec.id, equals('bge-small-en-v1.5'));
      expect(spec.meta['dimensions'], equals(384));
    });

    test('lookup throws ArgumentError for an unregistered model id', () {
      expect(() => ModelCatalog.lookup('not-a-model'), throwsArgumentError);
    });

    test('lookup throws UnsupportedError for an unvalidated model', () {
      expect(() => ModelCatalog.lookup('bge-m3-v1.0'), throwsUnsupportedError);
    });

    test('isKnown distinguishes registered from unknown ids', () {
      expect(ModelCatalog.isKnown('bge-small-en-v1.5'), isTrue);
      expect(ModelCatalog.isKnown('bge-m3-v1.0'), isTrue);
      expect(ModelCatalog.isKnown('not-a-model'), isFalse);
    });

    test('defaultModelId is bge-small-en-v1.5', () {
      expect(ModelCatalog.defaultModelId, equals('bge-small-en-v1.5'));
    });

    test('all returns a non-empty iterable', () {
      expect(ModelCatalog.all, isNotEmpty);
    });
  });

  group('quantise / dequantise', () {
    test(
      'round-trips a synthesised L2-normalised vector within SQ8 tolerance',
      () {
        // Build a 384-element vector on the unit sphere.
        final raw = List.generate(384, (i) => cos(i * 0.1));
        final norm = sqrt(raw.fold<double>(0.0, (s, v) => s + v * v));
        final normalised = Float32List.fromList(
          raw.map((v) => v / norm).toList(),
        );

        final quantised = quantise(normalised);
        expect(quantised.length, equals(384));

        final recovered = dequantise(quantised);
        expect(recovered.length, equals(384));

        // Per-element error is bounded by one SQ8 step (2.0/255 ≈ 0.00784).
        for (var i = 0; i < normalised.length; i++) {
          expect(recovered[i], closeTo(normalised[i], 0.009));
        }
      },
    );
  });

  // ── Native ORT + full embedding pipeline ──────────────────────────────────
  // Downloads BGE Small En v1.5 (~127 MB) on first run; subsequent runs use
  // the cache in modelCache.  Allow up to 10 minutes for a cold-cache run.
  //
  // What these tests verify:
  //   - The betto_onnxrt native-assets hook staged the ORT binary correctly.
  //   - OnnxRuntime.load() initialises the native library.
  //   - The full tokenise → infer → pool → L2-normalise pipeline produces
  //     geometrically sensible outputs.

  group('OnnxEmbeddingModel', () {
    OnnxEmbeddingModel? model;

    setUpAll(() async {
      model = await OnnxEmbeddingModel.load(
        cacheDir: modelCache.path,
        onProgress: (received, total) =>
            debugPrint('Model download: $received / $total bytes'),
      );
    });

    tearDownAll(() => model?.dispose());

    test('reports correct model identity and output dimensions', () {
      expect(model!.modelId, equals('bge-small-en-v1.5'));
      expect(model!.dimensions, equals(384));
    });

    test('embed returns a 384-element unit-norm float32 vector', () async {
      final (embedding, _) = await model!.embed('hello world');
      expect(embedding.length, equals(384));
      final l2sq = embedding.fold<double>(0.0, (s, v) => s + v * v);
      expect(sqrt(l2sq), closeTo(1.0, 1e-5));
    });

    test(
      'semantically similar pairs score higher than dissimilar pairs',
      () async {
        final (a, _) = await model!.embed('The cat sat on the mat');
        final (b, _) = await model!.embed('A cat rested on a rug');
        final (c, _) = await model!.embed(
          'Stock market volatility increased sharply',
        );

        double dot(Float32List x, Float32List y) => Iterable.generate(
          x.length,
          (i) => x[i] * y[i],
        ).fold<double>(0.0, (s, v) => s + v);

        // a·b (semantically close) must exceed a·c (unrelated topic).
        expect(dot(a, b), greaterThan(dot(a, c)));
      },
    );

    test('embed sets truncated=true when text exceeds 510 tokens', () async {
      final longText = List.filled(600, 'word').join(' ');
      final (embedding, truncated) = await model!.embed(longText);
      expect(embedding.length, equals(384));
      expect(truncated, isTrue);
    });

    test('embed sets truncated=false for short text', () async {
      final (_, truncated) = await model!.embed('brief sentence');
      expect(truncated, isFalse);
    });

    test('embed handles empty string without throwing', () async {
      final (embedding, truncated) = await model!.embed('');
      expect(embedding.length, equals(384));
      expect(truncated, isFalse);
    });

    test(
      'SQ8 round-trip of a real embedding stays within per-element tolerance',
      () async {
        final (embedding, _) = await model!.embed(
          'test sentence for quantisation fidelity',
        );
        final recovered = dequantise(quantise(embedding));
        for (var i = 0; i < embedding.length; i++) {
          expect(recovered[i], closeTo(embedding[i], 0.009));
        }
      },
    );

    test(
      'consecutive embeds of the same text return identical vectors',
      () async {
        const text = 'determinism check';
        final (first, _) = await model!.embed(text);
        final (second, _) = await model!.embed(text);
        for (var i = 0; i < first.length; i++) {
          expect(second[i], equals(first[i]));
        }
      },
    );
  });

  // ── XlmRobertaTokenizer parity gate ────────────────────────────────────────
  // Downloads the real `multilingual-e5-small` tokenizer.json (~17 MB) on
  // first run; subsequent runs use the cache in modelCache. This is a direct
  // HTTP fetch (not ModelDownloader/ModelCatalog — multilingual-e5-small is
  // not yet a registered catalog model; that is a separate, later embedding
  // model work item) with no checksum verification, since this test only
  // needs *a* copy of the real vocabulary to exercise byte-exact parity, not
  // model-distribution-grade integrity guarantees.
  //
  // What this verifies:
  //   - XlmRobertaTokenizer.load() correctly extracts and parses the real
  //     precompiled_charsmap trie and the real 250k-entry Unigram vocabulary.
  //   - encode() reproduces real HuggingFace AutoTokenizer output byte-exactly
  //     on all 11 smoke-corpus entries (ar, edge_fullwidth, edge_nfd,
  //     edge_punct, en, hi, ko, ru, th, vi, zh) -- the same gate Phase 0's
  //     spike passed, now exercised against production code.
  //   - Padding, attention mask, and truncation behave correctly against the
  //     real vocabulary (pad id 1, not BERT's 0).
  //
  // This intentionally reintroduces a huggingface.co network dependency at
  // test time -- an accepted, explicitly-stated divergence from this
  // package's usual "commit small fixtures, run fully offline" preference,
  // because committing the full 17 MB file (98%+ an unused-for-this-purpose
  // 250k-entry vocab table) is worse. See
  // `plan_0_06_wi11_xlmr_tokenizer.md`'s Phase 1 "Tests" section.

  group('XlmRobertaTokenizer (multilingual-e5-small)', () {
    late XlmRobertaTokenizer tokenizer;
    late Map<String, dynamic> smokeCorpus;
    late Map<String, dynamic> referenceIds;

    setUpAll(() async {
      final tokenizerCacheDir = Directory(
        '${modelCache.path}/multilingual-e5-small',
      )..createSync(recursive: true);
      final tokenizerJsonPath = '${tokenizerCacheDir.path}/tokenizer.json';
      final tokenizerJsonFile = File(tokenizerJsonPath);

      // Download once; reuse the cache on subsequent runs. A basic size
      // sanity check catches a truncated download or an HTML error page
      // masquerading as a 200 response (the real file is ~17 MB).
      if (!tokenizerJsonFile.existsSync() ||
          tokenizerJsonFile.lengthSync() < 1000000) {
        debugPrint('Downloading multilingual-e5-small tokenizer.json...');
        final client = HttpClient();
        try {
          final request = await client.getUrl(
            Uri.parse(
              'https://huggingface.co/intfloat/multilingual-e5-small/'
              'resolve/main/tokenizer.json',
            ),
          );
          final response = await request.close();
          if (response.statusCode != 200) {
            throw HttpException(
              'Failed to download tokenizer.json: '
              'HTTP ${response.statusCode}',
            );
          }
          await response.pipe(tokenizerJsonFile.openWrite());
        } finally {
          client.close();
        }
      }

      tokenizer = await XlmRobertaTokenizer.load(tokenizerJsonPath);

      // The smoke-corpus fixtures are committed under the package's own
      // test/fixtures/ and symlinked into this app's own
      // integration_test_app/test/fixtures/ (declared as Flutter `assets` in
      // pubspec.yaml) so there is a single source of truth. They must be
      // loaded via the Flutter asset bundle (rootBundle), not a host
      // filesystem path relative to Directory.current: on iOS/Android the
      // app runs inside a simulator/device sandbox with no view of the host
      // source checkout at all, so a `dart:io` File read of a relative host
      // path only ever worked on desktop (macOS/Linux/Windows), where the
      // host filesystem happens to be visible.
      smokeCorpus =
          jsonDecode(
                await rootBundle.loadString(
                  'test/fixtures/xlmr_smoke_corpus.json',
                ),
              )
              as Map<String, dynamic>;
      referenceIds =
          jsonDecode(
                await rootBundle.loadString(
                  'test/fixtures/xlmr_reference_ids.json',
                ),
              )
              as Map<String, dynamic>;
    });

    test('achieves byte-exact parity with real AutoTokenizer output on all '
        '11 smoke-corpus entries', () {
      for (final key in referenceIds.keys) {
        final text = smokeCorpus[key] as String;
        final expected = (referenceIds[key] as List).cast<int>();
        final output = tokenizer.encode(text);

        final actualContentIds = output.inputIds.take(expected.length).toList();
        expect(
          actualContentIds,
          equals(expected),
          reason: 'token id mismatch for smoke-corpus entry "$key"',
        );

        // Everything after the real content must be <pad> (id 1), not
        // BertTokenizer's [PAD] (id 0).
        for (var i = expected.length; i < output.inputIds.length; i++) {
          expect(
            output.inputIds[i],
            equals(1),
            reason: '$key: expected <pad> (1) at padded index $i',
          );
          expect(output.attentionMask[i], equals(0));
        }
      }
    });

    test('output arrays are exactly 512 (default maxLength) elements', () {
      final output = tokenizer.encode(smokeCorpus['en'] as String);
      expect(output.inputIds.length, equals(512));
      expect(output.attentionMask.length, equals(512));
      expect(output.tokenTypeIds.length, equals(512));
    });

    test('tokenTypeIds are all zeros (XLM-RoBERTa has no segment ids)', () {
      final output = tokenizer.encode(smokeCorpus['en'] as String);
      expect(output.tokenTypeIds.every((v) => v == 0), isTrue);
    });

    test('none of the smoke-corpus entries are truncated (all well under 512 '
        'tokens)', () {
      for (final key in referenceIds.keys) {
        final text = smokeCorpus[key] as String;
        final output = tokenizer.encode(text);
        expect(
          output.truncated,
          isFalse,
          reason: '$key unexpectedly reported truncated',
        );
      }
    });

    test('text exceeding 512 tokens sets truncated=true', () {
      final longText = List.filled(600, 'hello').join(' ');
      final output = tokenizer.encode(longText);
      expect(output.truncated, isTrue);
      expect(output.inputIds.length, equals(512));
    });
  });
}
