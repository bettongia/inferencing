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
// One-off dev tool (output committed, not run by CI) that builds the *text*
// half of the XLM-RoBERTa tokenizer byte-exact parity corpus (WI-4 Phase 1,
// see `docs/plans/plan_0_06_wi4_multilingual_embedding_model.md` in the
// `kmdb` repo). It downloads the same NLTK `udhr2`/`udhr` corpus
// `betto_lang_detector`'s `tool/generate_ngram_profiles.dart` already uses
// (see that file for the corpus's own provenance/licensing rationale) and
// extracts each language's Article 1 excerpt into a checked-in JSON text
// fixture.
//
// This script deliberately produces only *half* of the final parity
// fixture: the input text. The other half -- expected token ids from
// HuggingFace's `AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')`
// -- requires a Python environment this project's tooling does not have, and
// is a separate, manual hand-off step (see the plan's Phase 1). Once that
// step produces expected ids, they get merged with this script's output into
// `test/fixtures/xlmr_parity_corpus.json`, the final, permanently-static
// fixture the byte-exact parity test asserts against.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:http/http.dart' as http;

const _cacheDir = 'tool/.cache';
const _outputPath = 'test/fixtures/xlmr_parity_corpus_text.json';

/// The primary corpus: NLTK's `udhr2` package, a clean UTF-8 re-encoding of
/// the Universal Declaration of Human Rights translations. Archive members
/// live at `udhr2/{code}.txt`. Identical URL to `betto_lang_detector`'s own
/// copy of this constant -- see that package's `tool/generate_ngram_profiles.dart`
/// for the corpus's own provenance write-up.
const _udhr2Url =
    'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/udhr2.zip';

/// Fallback corpus: NLTK's original `udhr` package, needed only for Kurdish
/// (Kurmanji, Latin script) -- `udhr2` only has the Arabic-script Sorani
/// variant. See `betto_lang_detector`'s copy of this constant for the same
/// rationale; this tool needs the identical fallback since it reuses
/// [_corpusMember] verbatim.
const _udhrUrl =
    'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/udhr.zip';

/// Language code -> archive member path.
///
/// Copied verbatim from `betto_lang_detector`'s `tool/generate_ngram_profiles.dart`
/// (`_corpusMember`) rather than depending on that package -- pulling in a
/// sibling application package just for one small constant map would be a
/// much larger, wrong-direction dependency than duplicating ~60 lines of
/// data. Per that file's own doc comment, per-language script/orthography
/// choices recorded there are judgment calls, not derivable from the data
/// itself; kept unchanged here so this tool's corpus selection doesn't
/// silently diverge from the sibling repo's.
const Map<String, String> _corpusMember = {
  'af': 'udhr2/afr.txt',
  'ar': 'udhr2/arb.txt', // Standard Arabic, among several regional variants
  'bg': 'udhr2/bul.txt',
  'bn': 'udhr2/ben.txt',
  'br': 'udhr2/bre.txt',
  'ca': 'udhr2/cat.txt',
  'cs': 'udhr2/ces.txt',
  'da': 'udhr2/dan.txt',
  'de': 'udhr2/deu.txt',
  'el': 'udhr2/ell_monotonic.txt', // modern standard orthography
  'en': 'udhr2/eng.txt',
  'eo': 'udhr2/epo.txt',
  'es': 'udhr2/spa.txt',
  'et': 'udhr2/est.txt',
  'eu': 'udhr2/eus.txt',
  'fa': 'udhr2/pes_1.txt', // Iranian Persian, first of two available texts
  'fi': 'udhr2/fin.txt',
  'fr': 'udhr2/fra.txt',
  'ga': 'udhr2/gle.txt',
  'gl': 'udhr2/glg.txt',
  'gu': 'udhr2/guj.txt',
  'ha': 'udhr2/hau_NG.txt', // Nigeria orthography (larger speaker population)
  'he': 'udhr2/heb.txt',
  'hi': 'udhr2/hin.txt',
  'hr': 'udhr2/hrv.txt',
  'hu': 'udhr2/hun.txt',
  'hy': 'udhr2/hye.txt',
  'id': 'udhr2/ind.txt',
  'it': 'udhr2/ita.txt',
  'ja': 'udhr2/jpn.txt',
  'ko': 'udhr2/kor.txt',
  'ku': 'udhr/Kurdish-UTF8', // Kurmanji, Latin script -- see [_udhrUrl]
  'la': 'udhr2/lat.txt',
  'lt': 'udhr2/lit.txt',
  'lv': 'udhr2/lav.txt',
  'mr': 'udhr2/mar.txt',
  'ms': 'udhr2/mly_latn.txt', // Latin script, as opposed to Jawi (Arabic)
  'nl': 'udhr2/nld.txt',
  'no': 'udhr2/nob.txt', // Bokmal, the majority written standard
  'pl': 'udhr2/pol.txt',
  'pt': 'udhr2/por_PT.txt', // European Portuguese, among two regional texts
  'ro': 'udhr2/ron.txt',
  'ru': 'udhr2/rus.txt',
  'sk': 'udhr2/slk.txt',
  'sl': 'udhr2/slv.txt',
  'so': 'udhr2/som.txt',
  'st': 'udhr2/sot.txt',
  'sv': 'udhr2/swe.txt',
  'sw': 'udhr2/swh.txt',
  'th': 'udhr2/tha.txt',
  'tl': 'udhr2/tgl.txt',
  'tr': 'udhr2/tur.txt',
  'uk': 'udhr2/ukr.txt',
  'ur': 'udhr2/urd.txt',
  'vi': 'udhr2/vie.txt',
  'yo': 'udhr2/yor.txt',
  'zh': 'udhr2/cmn_hans.txt', // Simplified script, among two script variants
  'zu': 'udhr2/zul.txt',
};

/// Hand-authored edge cases the UDHR corpus itself can't cover naturally
/// (WI-4 Phase 1's second checklist item): an empty string, a very long
/// string near `multilingual-e5-small`'s ~512-token limit, and a string
/// mixing several scripts in one input.
///
/// The "near 512 tokens" entry ([_buildLongEdgeCase]) is necessarily a
/// length *estimate* -- this tool cannot run the real tokenizer to count
/// tokens exactly (that is the whole reason Phase 1 has a Python hand-off).
/// It targets ~380 English words based on the observed ~1.3
/// tokens-per-word ratio for plain English text in this vocabulary (see the
/// Phase 0 spike's `en` entry: 30 words -> 39 ids including `<s>`/`</s>`).
/// Once the real token ids come back, if this proves off-target it can be
/// resized -- it does not need to be exact, only "near the limit" per the
/// plan.
Map<String, String> _edgeCases(String englishFullText) => {
  'edge_empty': '',
  'edge_long_near_512_tokens': _buildLongEdgeCase(englishFullText),
  'edge_mixed_script':
      'Hello world! مرحبا بالعالم 你好，世界 Привет, мир! '
      'こんにちは世界 नमस्ते दुनिया',
};

/// Target word count for [_buildLongEdgeCase] -- see [_edgeCases]'s doc
/// comment for the ratio this is based on.
const _longEdgeCaseTargetWords = 380;

Future<void> main() async {
  Directory(_cacheDir).createSync(recursive: true);

  final udhr2Archive = await _fetchArchive('udhr2.zip', _udhr2Url);
  final udhrArchive = await _fetchArchive('udhr.zip', _udhrUrl);

  final corpus = <String, String>{};
  final skipped = <String>[];

  for (final entry in _corpusMember.entries) {
    final code = entry.key;
    final memberPath = entry.value;
    final isUdhr2 = memberPath.startsWith('udhr2/');
    final archive = isUdhr2 ? udhr2Archive : udhrArchive;

    final file = archive.findFile(memberPath);
    if (file == null) {
      skipped.add('$code ($memberPath): not found in archive');
      continue;
    }
    final text = utf8.decode(file.readBytes()!, allowMalformed: true);
    final article1 = isUdhr2
        ? _extractArticle1FromUdhr2(text)
        : _extractArticle1FromLegacyUdhr(text);
    if (article1 == null || article1.trim().isEmpty) {
      skipped.add('$code ($memberPath): Article 1 extraction failed');
      continue;
    }
    corpus[code] = article1;
  }

  if (skipped.isNotEmpty) {
    // Non-fatal by design: the plan's own Phase 1 wording asks for "as many
    // of its ~52 mapped languages as practical", not all-or-nothing --
    // unlike generate_ngram_profiles.dart's strict all-or-nothing check,
    // a handful of extraction misses shouldn't block the rest of the
    // corpus from being generated. Printed loudly so misses aren't silent.
    stderr.writeln('Skipped ${skipped.length} language(s):');
    for (final reason in skipped) {
      stderr.writeln('  - $reason');
    }
  }

  // A rough plausibility check on every successful extraction: Article 1 is
  // always a short, 1-3 sentence paragraph across every UDHR translation, so
  // an extraction far outside that range is more likely a heuristic
  // misfire (e.g. grabbing the preamble instead) than a genuinely terse or
  // verbose translation. Printed as warnings, not treated as failures --
  // this tool cannot know for certain without a native-language reader, so
  // it surfaces the signal rather than silently dropping the entry.
  const plausibleMin = 20;
  const plausibleMax = 900;
  for (final entry in corpus.entries) {
    final len = entry.value.length;
    if (len < plausibleMin || len > plausibleMax) {
      stderr.writeln(
        'Warning: ${entry.key} extracted length $len chars is outside the '
        'plausible Article-1 range [$plausibleMin, $plausibleMax] -- '
        'spot-check this entry by hand: ${entry.value}',
      );
    }
  }

  final englishFullText = utf8.decode(
    udhr2Archive.findFile('udhr2/eng.txt')!.readBytes()!,
  );
  corpus.addAll(_edgeCases(englishFullText));

  _writeCorpus(corpus);
}

/// Downloads (or reuses a cached copy of) the zip at [url], decoding it as
/// an in-memory [Archive] for random-access member lookup. Identical
/// caching strategy to `betto_lang_detector`'s `_fetchArchive`.
Future<Archive> _fetchArchive(String cacheFileName, String url) async {
  final cacheFile = File('$_cacheDir/$cacheFileName');
  Uint8List bytes;
  if (cacheFile.existsSync()) {
    // ignore: avoid_print
    print('Using cached archive: ${cacheFile.path}');
    bytes = cacheFile.readAsBytesSync();
  } else {
    // ignore: avoid_print
    print('Downloading: $url');
    final response = await http.get(Uri.parse(url));
    if (response.statusCode != 200) {
      throw StateError('Failed to download $url: HTTP ${response.statusCode}');
    }
    bytes = response.bodyBytes;
    cacheFile.writeAsBytesSync(bytes);
  }
  return ZipDecoder().decodeBytes(bytes);
}

/// Extracts Article 1's paragraph text from a `udhr2`-format document.
///
/// `udhr2` documents are laid out as a sequence of blank-line-separated
/// blocks: a title (sometimes merged with a translated "Preamble" heading),
/// the preamble body (always the longest block near the top of the
/// document -- a list of "Whereas..."-style clauses), then one block per
/// numbered article (heading line + one-to-few sentence paragraph).
///
/// Rather than pattern-matching the (per-language, translated) heading text
/// -- which would need a translation table this tool doesn't have -- this
/// locates the preamble structurally: it is reliably the longest of the
/// first few blocks, regardless of whether the title and preamble heading
/// happen to be merged into one block or split across two. Article 1 is
/// simply the block immediately following it. Verified by hand against ~15
/// languages spanning Latin, Arabic, Cyrillic, CJK, Devanagari, Thai,
/// Hebrew, and Greek scripts during this tool's development -- see the
/// plan's Phase 1 notes for specifics.
String? _extractArticle1FromUdhr2(String text) {
  final blocks = _splitIntoBlankLineBlocks(text);
  if (blocks.isEmpty) return null;

  // Only the first few blocks are ever title/preamble material -- bounding
  // the search window avoids misfiring on a later article that happens to
  // be unusually long.
  const searchWindow = 4;
  var anchorIndex = 0;
  var maxLen = -1;
  for (var i = 0; i < blocks.length && i < searchWindow; i++) {
    final len = blocks[i].join(' ').length;
    if (len > maxLen) {
      maxLen = len;
      anchorIndex = i;
    }
  }

  final articleIndex = anchorIndex + 1;
  if (articleIndex >= blocks.length) return null;
  final block = blocks[articleIndex];
  if (block.isEmpty) return null;
  // First line is the (translated) "Article 1" heading; the remainder is
  // the paragraph body. A single-line block (unexpected but not impossible)
  // falls back to using the whole thing.
  return block.length > 1 ? block.skip(1).join(' ') : block.first;
}

/// Splits [text] into blocks of non-blank lines, separated by one or more
/// blank (or whitespace-only) lines.
List<List<String>> _splitIntoBlankLineBlocks(String text) {
  final blocks = <List<String>>[];
  var current = <String>[];
  for (final rawLine in text.split('\n')) {
    final line = rawLine.trim();
    if (line.isEmpty) {
      if (current.isNotEmpty) {
        blocks.add(current);
        current = [];
      }
    } else {
      current.add(line);
    }
  }
  if (current.isNotEmpty) blocks.add(current);
  return blocks;
}

/// Extracts Article 1's paragraph text from a legacy `udhr`-format document
/// (only [_corpusMember]'s `ku` entry uses this archive).
///
/// Unlike `udhr2`, these older files have no blank-line separators at all --
/// every heading and paragraph is its own single line, back-to-back. Instead
/// this finds the *second* short line (a heading is a handful of characters,
/// e.g. "Dîbaçe"/"Bend 1"; every preamble/article sentence is far longer)
/// and returns the line right after it -- the first short line is the
/// preamble heading, the second is Article 1's.
String? _extractArticle1FromLegacyUdhr(String text) {
  final lines = text
      .split('\n')
      .map((l) => l.trim())
      .where((l) => l.isNotEmpty)
      .toList();

  const headingMaxLength = 20;
  final shortLineIndices = <int>[
    for (var i = 0; i < lines.length; i++)
      if (lines[i].length <= headingMaxLength) i,
  ];
  if (shortLineIndices.length < 2) return null;

  final article1HeadingIndex = shortLineIndices[1];
  final contentIndex = article1HeadingIndex + 1;
  if (contentIndex >= lines.length) return null;
  return lines[contentIndex];
}

/// Builds the "near the 512-token limit" edge case by concatenating whole
/// paragraphs from the full English UDHR text (title, preamble, then
/// articles in order) until reaching [_longEdgeCaseTargetWords] words -- see
/// [_edgeCases]'s doc comment for why this is a length estimate, not an
/// exact token-count target.
String _buildLongEdgeCase(String englishFullText) {
  final blocks = _splitIntoBlankLineBlocks(englishFullText);
  final buffer = StringBuffer();
  var wordCount = 0;
  for (final block in blocks) {
    final blockText = block.join(' ');
    if (wordCount >= _longEdgeCaseTargetWords) break;
    buffer.write('$blockText ');
    wordCount += blockText.split(RegExp(r'\s+')).length;
  }
  return buffer.toString().trim();
}

/// Writes the extracted-text corpus (UDHR excerpts + edge cases) to
/// [_outputPath] as pretty-printed JSON, sorted by key for a stable diff.
///
/// This is deliberately *not* the final parity fixture
/// (`test/fixtures/xlmr_parity_corpus.json`) -- it holds only the input text
/// half; the expected-token-ids half requires a manual Python step (see this
/// file's top-level doc comment).
void _writeCorpus(Map<String, String> corpus) {
  final sortedKeys = corpus.keys.toList()..sort();
  final sorted = {for (final key in sortedKeys) key: corpus[key]};

  const encoder = JsonEncoder.withIndent('  ');
  File(_outputPath).writeAsStringSync('${encoder.convert(sorted)}\n');
  // ignore: avoid_print
  print('Generated: $_outputPath (${corpus.length} entries)');
}
