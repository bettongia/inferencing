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

// ignore_for_file: avoid_print

import 'package:betto_inferencing/betto_inferencing.dart';

/// Example demonstrating [ModelCatalog] — the allowlist of supported embedding
/// models.
///
/// Covers:
///   - Listing all registered models and their metadata
///   - Retrieving the default / recommended model
///   - Checking registration status with [ModelCatalog.isKnown]
///   - Error handling for unvalidated and unknown model IDs
///
/// No model download or ORT session is required. Run with:
///   dart run example/model_catalog.dart
void main() {
  // ── 1. List all registered models ─────────────────────────────────────────
  // ModelCatalog.all returns every registered ModelSpec, validated or not.
  // Useful for building a CLI model-selector or admin tool.
  print('=== Registered models ===');
  for (final spec in ModelCatalog.all) {
    final dims = spec.meta['dimensions'];
    print('  ${spec.id}  (dimensions: $dims)');
  }
  print('');

  // ── 2. Default / recommended model ────────────────────────────────────────
  // ModelCatalog.defaultModelId is the stable ID of the production-ready model
  // to use when no explicit choice is made.
  final defaultId = ModelCatalog.defaultModelId;
  print('Default model: $defaultId');

  final spec = ModelCatalog.lookup(defaultId);
  print('  id         : ${spec.id}');
  print('  dimensions : ${spec.meta['dimensions']}');
  print('  file keys  : ${spec.files.keys.join(', ')}');
  print('');

  // ── 3. Registration check ─────────────────────────────────────────────────
  // ModelCatalog.isKnown returns true for any registered ID regardless of
  // validation status. Use it to detect stale config files that reference a
  // known but unvalidated model.
  print('=== isKnown checks ===');
  for (final id in [
    'bge-small-en-v1.5', // validated, production-ready
    'multilingual-e5-small', // validated, production-ready (multilingual)
    'placeholder-model', // permanent test fixture, never validated
    'my-custom-model', // not registered at all
  ]) {
    print('  isKnown("$id"): ${ModelCatalog.isKnown(id)}');
  }
  print('');

  // ── 4. Error: unvalidated model ───────────────────────────────────────────
  // placeholder-model is registered in the catalog but lookup() always
  // throws UnsupportedError — it is a permanent test fixture, not a model
  // pending validation.
  print('=== Error handling ===');
  try {
    ModelCatalog.lookup('placeholder-model');
  } on UnsupportedError catch (e) {
    print('UnsupportedError for unvalidated model (expected):');
    print('  $e');
  }
  print('');

  // ── 5. Error: unknown model ───────────────────────────────────────────────
  // Any ID not present in the catalog throws ArgumentError immediately.
  try {
    ModelCatalog.lookup('my-custom-model');
  } on ArgumentError catch (e) {
    print('ArgumentError for unknown model (expected):');
    print('  $e');
  }
}
