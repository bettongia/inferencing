# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## General

Work is planned using specifications in the `docs/plans` directory. When working
on plans make sure you review `docs/plans/README.md` file for guidance. When
asked to plan something do not commence implementation until explicitly told to
do so.

The `docs/roadmap` directory is used to track future work items and their
priority. This is worth reviewing when working on the codebase as current work
may intersect with the roadmap.

We'll create plans for our work and place them in the `docs/plans/` directory.
When the planned work has been completed we'll move them to
`docs/plans/completed`.

Quality assurance is critical to this project and you need to maintain a minimum
of 90% test coverage at all times. You must also run all tests successfully
before considering a task to be complete.

Consider edge-cases and failure scenarios when preparing tests - it is critical
not just to focus on easy, "golden-path" tests.

All public classes, methods and properties must have appropriate doc comments.
You may include examples in doc comments if you believe it will help another
developer.

Any complex segments of code should be commented so as to describe the process
and rationale for the approach.

All code files must have a license at the top. The template file is
@header_template.txt. You must add the comment syntax appropriate to the
programming language. Also replace `{{.Year}}` to match the current year.

## Repository Layout

```
lib/
  betto_inferencing.dart        # top-level library — public API exports
  src/
    embedding_model.dart        # EmbeddingModel abstract interface
    onnx_embedding_model.dart   # OnnxEmbeddingModel (ONNX Runtime implementation)
    bert_tokenizer.dart         # BertTokenizer + TokenizerOutput
    model_catalog.dart          # ModelCatalog (AllowlistProvider)
    sq8.dart                    # quantise / dequantise helpers
    math_utils.dart             # internal: meanPool, l2Normalize, cosineSimilarity
test/
  bert_tokenizer_test.dart
  betto_inferencing_test.dart
  math_utils_test.dart
  model_catalog_test.dart
  model_downloader_test.dart
  sq8_test.dart
example/
  example.dart                  # usage example (download-on-demand)
docs/
  index.md                      # site landing page
  spec/README.md                # technical specification
  roadmap/v0.md                 # v0 roadmap
  plans/                        # implementation plans
  reviews/                      # code review notes
site/                           # generated HTML (pandoc + dart doc); do not edit
```

## Commands

The `Makefile` should contain all key development lifecycle commands. In
general, `make` should be preferred to directly running commands such as `dart`
and `flutter`.

```bash
# Run tests
make test

# Analyze/lint
make analyze

# Format code
make format

# Check formatting without modifying files
make format_check

# Coverage report (outputs to site/coverage/)
make coverage

# Build docs site — spec, roadmap, API ref (requires pandoc and dart doc)
make doc_site

# Add missing license headers
make license_add

# Check license headers
make license_check

# Install dependencies and activate coverage tool
make prepare

# Run checks before committing code
make pre_commit

# Full pipeline: clean, prepare, license_check, format, analyze, test, coverage, doc_site
make
```

## Implementation Status

v0.1.0-dev.1 core feature set is complete:

- `EmbeddingModel` — abstract interface for text-to-vector embedding
- `OnnxEmbeddingModel` — ONNX Runtime implementation backed by BGE Small En v1.5
- `BertTokenizer` + `TokenizerOutput` — BERT WordPiece tokenizer
- `ModelCatalog` — allowlist with download-on-demand gating
- `quantise` / `dequantise` — SQ8 scalar quantization helpers

`bge-m3-v1.0` is registered in `ModelCatalog` but not yet validated; it will
throw `UnsupportedError` if passed to `ModelCatalog.lookup`.

## Architecture

The package is a thin layer between the consuming application and
`betto_onnxrt` (the ORT FFI binding):

```
consuming layer (betto_db / app)
  → EmbeddingModel interface only (no FFI dependency)
betto_inferencing
  → OnnxEmbeddingModel, BertTokenizer, ModelCatalog, SQ8
betto_onnxrt
  → OnnxRuntime FFI, ModelDownloader
```

Word segmentation inside `BertTokenizer` delegates to the `betto_lexical`
`Tokenizer` interface (default: `RegExpTokenizer`; can substitute
`IcuTokenizer` from `package:betto_icu`).

See [docs/spec/README.md](docs/spec/README.md) for the full specification
including the embedding pipeline, API contracts, and platform support matrix.

## Documentation

Full specification is in [docs/spec/](docs/spec/) (Pandoc Markdown). The built
HTML lives in [site/](site/) and is generated via `make doc_site`. Key docs:

- [docs/spec/README.md](docs/spec/README.md) — technical specification
  (architecture, public API, data flow, error handling, platform support)
- [docs/roadmap/v0.md](docs/roadmap/v0.md) — v0 roadmap and planned work
