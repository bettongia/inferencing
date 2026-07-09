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

/// ONNX Runtime inference and embedding models for dense text retrieval.
///
/// Provides [OnnxEmbeddingModel] (implements [EmbeddingModel]) backed by
/// either BGE Small En v1.5 or `multilingual-e5-small` via the
/// `betto_onnxrt` [OnnxRuntime] API, a [ModelTokenizer] abstraction over
/// [BertTokenizer] (BERT WordPiece) and [XlmRobertaTokenizer]
/// (XLM-RoBERTa-family SentencePiece/Unigram), an [EmbeddingKind] parameter
/// on [EmbeddingModel.embed] distinguishing indexing- from query-time text,
/// [quantise]/[dequantise] helpers for SQ8 vector quantisation, and a
/// [ModelCatalog] of supported models with download-on-demand via
/// [ModelDownloader] from `betto_onnxrt`.
///
/// ## Platform support
///
/// This package is **native-only** (macOS arm64, Linux, Windows, Android,
/// iOS). It must not be imported on the web platform.
///
/// **iOS** requires the `betto_onnxrt_ios` companion Flutter plugin (iOS ≥ 16),
/// which statically links the ORT XCFramework via SPM. Add it alongside
/// `betto_inferencing` in your Flutter app's `pubspec.yaml`:
///
/// ```yaml
/// dependencies:
///   betto_inferencing: ^0.1.0-dev.1
///   betto_onnxrt_ios: ^0.1.0-dev.2   # iOS only
/// ```
///
/// ## ORT binary acquisition
///
/// The ONNX Runtime shared library is staged at build time by the
/// `betto_onnxrt` native-assets build hook (`hook/build.dart` in the
/// `betto_onnxrt` package). The hook downloads and SHA-256-verifies the
/// platform-appropriate ORT binary from the official Microsoft ORT GitHub
/// Releases. On Android the `.so` is bundled by the build system.
library;

// Re-export betto_onnxrt types that callers of betto_inferencing need directly.
// ModelSpec, ModelFile, ModelDownloader, ResolvedModel, and DownloadProgress
// are part of the stable public surface of this package.
export 'package:betto_onnxrt/betto_onnxrt.dart'
    show DownloadProgress, ModelDownloader, ModelFile, ModelSpec, ResolvedModel;

export 'src/bert_tokenizer.dart' show BertTokenizer, TokenizerOutput;
export 'src/embedding_model.dart' show EmbeddingKind, EmbeddingModel;
export 'src/model_catalog.dart' show ModelCatalog;
export 'src/model_tokenizer.dart' show ModelTokenizer;
export 'src/onnx_embedding_model.dart' show OnnxEmbeddingModel;
export 'src/sq8.dart' show quantise, dequantise;
export 'src/xlmr_tokenizer.dart' show XlmRobertaTokenizer;
