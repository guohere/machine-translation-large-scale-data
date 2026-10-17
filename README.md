# Machine Translation at Scale: Data Processing, Synthetic Data and Preference Tuning

An end-to-end post-training pipeline for English→German machine translation with a small LLM. Its focus is
the parts that decide quality in production: cleaning large corpora, generating and filtering synthetic
translations, reward-based fine-tuning, and evaluation that can gate a release.

| # | Stage | Status |
|---|---|---|
| 1 | **Large-scale corpus cleaning (PySpark):** normalisation, heuristic filters, language ID, exact and MinHash near-dedup | ✅ implemented, tested, benchmarked |
| 2 | **Synthetic data generation:** forward-translate monolingual English with a stronger open model (vLLM), filter with a reference-free quality-estimation (QE) model such as CometKiwi | planned |
| 3 | **Training:** SFT of Qwen2.5 on organic and synthetic bitext, then GRPO with a QE reward, compared with DPO on best-vs-worst sampled translations | planned |
| 4 | **Evaluation and release gating:** chrF, COMET and MetricX on FLORES-200 / WMT24++, a regression gate in CI, versioned checkpoints | planned |

## Stage 1: corpus cleaning with PySpark

```
read (tsv/text/parquet/jsonl) → normalise → heuristic filters → language ID → exact dedup → near dedup → parquet
```

```bash
pip install -r requirements.txt -e .          # needs Java 17 or 21
scripts/download_news_commentary.sh           # ~400k EN-DE pairs from statmt.org
python -m mtpipe.data.clean --config configs/news_commentary_en_de.yaml
```

The output directory contains the cleaned parquet (`data/`), and `stats.json` with rows in and out, rejections
by reason and wall time per stage. `rejected/<reason>.jsonl` holds a stable sample of rejected rows for every
filter, so each filter's precision can be checked by eye. All thresholds live in the YAML config.

### Filters

| Filter | Rejects | Why it matters for MT |
|---|---|---|
| Normalisation | (rewrites text) NFKC, invisible characters, whitespace | the same sentence typed two ways should filter and dedup identically |
| `empty`, `too_short`, `too_long` | word count outside `[3, 150]` on either side | fragments and run-on segments are usually alignment errors |
| `few_letters` | under 50% letters | tables, numbers and code |
| `length_ratio` | character ratio > 2.5 between sides | the classic sign of a misaligned pair |
| `copy` | source equals target | untranslated copies teach the model to echo its input |
| `wrong_lang_*`, `unsure_lang_*` | language ID (py3langid, offline) disagrees, or confidence < 0.5 | wrong-language targets are a major source of off-target translations |
| exact dedup | case-insensitive duplicate source sentences | wasted compute, over-weighted boilerplate |
| near dedup | ~0.79+ Jaccard on character 5-grams | near-copies that differ only in punctuation, case or a few words |

### Design notes

- **Filters run in the JVM where possible.** Length, ratio, letter-share and copy checks are native Spark column
  expressions. Python runs only for Unicode normalisation and language ID, both vectorised with Arrow
  `pandas_udf`s. The language-ID model is loaded once per worker process, not once per batch.
- **One pass per stage, with a reason per row.** Each row is tagged with the first check it fails rather than
  being filtered step by step, so per-reason counts come from one aggregation. Language ID runs only on rows
  that pass the cheap filters. It is a separate step because Spark evaluates Python UDFs on every row even
  inside `CASE WHEN`.
- **Deterministic output.** Exact dedup keeps the lexicographically smallest row of each group, not whatever
  `dropDuplicates` happens to see first. Row ids are content hashes (`xxhash64`) rather than
  `monotonically_increasing_id()`, which can change when Spark recomputes a DataFrame.
- **Banded MinHash instead of Spark ML's `MinHashLSH`.** My first version used
  `MinHashLSH.approxSimilarityJoin`. On a 200k-row benchmark it had not finished the self-join after 10
  minutes. That API treats two rows as candidates when *any single* hash matches, so even rows that are only 30%
  similar collide 83% of the time with 5 hash tables, and the candidate join grows roughly quadratically. The
  replacement computes 100 MinHashes per row, groups them into 10 bands of 10 and keys a bucket by each band's
  hash. A pair with Jaccard similarity *s* shares a bucket with probability 1 − (1 − s¹⁰)¹⁰, a steep S-curve
  around 0.79. Within a bucket the smallest id is kept, using a window `min` instead of a pairwise join, so the
  cost is linear in rows. The same 200k rows then took about 33 s.
- **`.gz` input is one partition.** Gzip cannot be split, so the config repartitions after reading.
  Otherwise every filter would run on a single core.

### Benchmark

`scripts/make_benchmark_corpus.py` generates a corpus with a known amount of noise. The sentences are word
salad, useless for training, but the expected removals are known exactly. On 1,000,000 EN-DE pairs (187 MB),
`local[*]` on a 4-core, 16 GB cloud container:

| Stage | Removed | Injected | Time |
|---|---:|---:|---:|
| Heuristic filters | 49,982 | ~50,000 (5% noise) | 23.9 s |
| Language ID | 681 | 0 (false positives on word salad) | 23.4 s |
| Exact dedup | 104,243 | ~100,000 (10%) | 4.7 s |
| Near dedup | 49,987 | ~50,000 (5%) | 29.6 s |
| Write parquet | | | 3.0 s |
| **Total** | **204,893 (79.5% kept)** | | **84.6 s** |

The near-dedup count matching the injected near-duplicates is the important check: banded MinHash is not
deleting distinct sentences. (Stage times are attributed to the action that triggers them, since Spark
evaluates lazily.)

> **To do:** run on News Commentary and a few million News Crawl sentences, and add the real-data numbers and
> rejection examples here.

## Tests

```bash
pip install -r requirements-dev.txt -e .
pytest
```

19 tests run against a local Spark session: normalisation edge cases, every filter with a row built to trip
it, language ID, exact and near dedup (including MinHash band properties), and an end-to-end run of the CLI
pipeline on a small gzipped TSV. CI runs them with `ruff` on every push.

## Layout

```
mtpipe/data/
  clean.py     pipeline orchestration and CLI
  config.py    YAML-backed configuration
  text.py      Unicode normalisation (plain Python, unit-tested)
  filters.py   heuristic and language filters as Spark column expressions
  langid.py    vectorised language-ID UDF
  dedup.py     exact dedup and banded MinHash near-dedup
configs/       one YAML file per corpus
scripts/       data download and benchmark corpus generation
tests/
```
