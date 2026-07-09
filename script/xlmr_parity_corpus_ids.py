#!/usr/bin/env python
# Copyright 2026 The Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# One-off manual step for WI-4 Phase 1's tokenizer parity corpus (see
# plan_0_06_wi4_multilingual_embedding_model.md in the kmdb repo). Not run by
# CI -- betto_inferencing's own tooling is pure Dart; this is the one step
# that genuinely needs a Python `transformers` environment (HuggingFace's
# `AutoTokenizer`), run manually once per corpus revision. Its output is
# merged by hand with the extracted-text fixture into the final, committed
# test/fixtures/xlmr_parity_corpus.json -- this script's own input and
# output files are intermediates, not committed on their own.
#
# Usage (from this directory):
#   dart run ../tool/generate_xlmr_parity_corpus.dart   # regenerates the
#                                                        # text input below
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   python3 xlmr_parity_corpus_ids.py

import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")

with open("../test/fixtures/xlmr_parity_corpus_text.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

ids = {}
for key, text in corpus.items():
    encoded = tokenizer(text, add_special_tokens=True, truncation=False, padding=False)
    ids[key] = encoded["input_ids"]

with open("../test/fixtures/xlmr_parity_corpus_ids.json", "w", encoding="utf-8") as f:
    json.dump(ids, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(ids)} entries to xlmr_parity_corpus_ids.json")
