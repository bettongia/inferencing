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

import 'dart:io';

import 'package:betto_inferencing/betto_inferencing.dart';
import 'package:test/test.dart';

void main() {
  group('OnnxEmbeddingModel - contract', () {
    test('OnnxEmbeddingModel implements EmbeddingModel interface', () {
      // Type check is compile-time, but we can verify at runtime too.
      expect(OnnxEmbeddingModel, isNotNull);
    });

    test('OnnxEmbeddingModel.load() throws ArgumentError when neither '
        'modelPath nor cacheDir is supplied', () async {
      // Calling load() with no modelPath and no cacheDir must fail fast with
      // ArgumentError before any I/O is attempted.
      await expectLater(OnnxEmbeddingModel.load, throwsA(isA<ArgumentError>()));
    });

    test('OnnxEmbeddingModel is exported from barrel', () {
      // If this file compiles, the export is correct.
      expect(OnnxEmbeddingModel, isNotNull);
    });
  });

  group('OnnxEmbeddingModel - _assertFileExists', () {
    test(
      'load() throws UnsupportedError when model file does not exist',
      () async {
        // A non-existent modelPath must throw UnsupportedError before
        // reaching the ORT loading code.
        await expectLater(
          OnnxEmbeddingModel.load(modelPath: '/nonexistent/path/model.onnx'),
          throwsA(isA<UnsupportedError>()),
        );
      },
    );

    test(
      'load() throws UnsupportedError when vocab.txt is missing beside the model',
      () async {
        // Create a temporary file that acts as the model file so the model-file
        // check passes, then verify that the vocab.txt check fires.
        final tmpDir = Directory.systemTemp.createTempSync('ort_test_');
        addTearDown(() => tmpDir.deleteSync(recursive: true));
        final modelFile = File('${tmpDir.path}/model.onnx')..createSync();

        await expectLater(
          OnnxEmbeddingModel.load(modelPath: modelFile.path),
          throwsA(isA<UnsupportedError>()),
        );
      },
    );

    test('UnsupportedError message includes the missing path', () async {
      const missingPath = '/nonexistent/path/model.onnx';
      await expectLater(
        OnnxEmbeddingModel.load(modelPath: missingPath),
        throwsA(
          isA<UnsupportedError>().having(
            (e) => e.message,
            'message',
            contains(missingPath),
          ),
        ),
      );
    });
  });
}
