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

import 'bert_tokenizer.dart' show TokenizerOutput;

/// Shared interface implemented by every tokenizer family
/// [OnnxEmbeddingModel] can select at runtime.
///
/// [BertTokenizer] and [XlmRobertaTokenizer] both implement this interface
/// and both already return the concrete [TokenizerOutput] type verbatim —
/// there is no separate `ModelInput` wrapper type. [OnnxEmbeddingModel.load]
/// picks which concrete tokenizer to construct based on
/// `ModelSpec.meta['tokenizerFamily']` (`'bert'` or `'xlmr'`), so
/// [OnnxEmbeddingModel.embed] has exactly one `_tokenizer.encode(text)` call
/// site regardless of which model family is loaded — registering a third
/// tokenizer family later only requires a new `ModelTokenizer` implementation
/// and a new `tokenizerFamily` value, not a change to [OnnxEmbeddingModel]
/// itself.
abstract interface class ModelTokenizer {
  /// Encodes [text] into a [TokenizerOutput] ready for ONNX inference.
  ///
  /// See the implementing class ([BertTokenizer] or [XlmRobertaTokenizer])
  /// for the exact special-token framing, padding id, and truncation
  /// behaviour — these differ between tokenizer families, but the shape of
  /// the returned [TokenizerOutput] does not.
  TokenizerOutput encode(String text);
}
