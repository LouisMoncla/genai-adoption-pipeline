# GenAI Adoption Pipeline — Code Explanation

> **LEGACY (as of 2026-07-17/23):** this document describes the original
> `main.py` / regression-based group-assignment / `aggregation.py` flow,
> which has been superseded by keyword-group matching against Domenico's
> validated keyword list (`pipeline/group_classification.py` +
> `keyword_lists/master_keywords.json`). Those old modules now live in
> `pipeline/legacy/` for provenance and are not part of the active pipeline.
> See the top-level `genai-adoption-pipeline/README.md` for how the current
> pipeline actually works. Kept here as historical/methodology reference.

This document explains the **raw pipeline code** in [`../pipeline/`]

It is split into two parts:

1. **Code Function Overview** — a step-by-step walk through the pipeline, starting from
   `config.yaml` and `main.py` and following the data through every stage. Read this first
   to get the shape of the whole thing.
2. **Detailed Deep Dive** — a function-by-function explanation of *how* each stage works
   (the filtering logic, the regression/group-assignment math, the flagging rules, the
   aggregations). Read this when you need to understand or modify a specific stage.

**One-line mental model:** raw postings → *clean & dedup by year* → *tag language* →
*translate keyword dictionary* → *(count hits → regression-classify keywords into groups →
measure co-occurrence/occupations → flag each ad)* → *aggregate flags into firm/time/industry
tables*. Each arrow is a folder under `output/`, and every stage is resumable because it
checks for its predecessor's output (and its own checkpoints) before doing any work.

**How to run it** (from the `genai-pipeline-handover/` root):

```bash
python -m pipeline.main                       # full run, uses pipeline/config.yaml by default
python -m pipeline.main --skip-to phase2      # resume from keyword scoring
python -m pipeline.main --config pipeline/config.yaml
```


> Final aggregation of **Phase III**, produces the parquet tables under `output/aggregation/`.

---

# Part 1 — Code Function Overview

The pipeline is a fixed sequence of stages. The orchestrator
([`pipeline/main.py`](../pipeline/main.py)) calls them in order, and each stage reads the
folder the previous stage wrote. That folder contract *is* the pipeline's data flow:

```
config.yaml ─▶ main.py ─▶ prepare_data ─▶ detect_languages ─▶ translate_keywords ─▶ score_postings ─▶ aggregate_results
                          (Phase I)        (language)          (translation)         (Phase II)        (Phase III)

Data/*.parquet
  └▶ output/prepared_data/year=YYYY/part.parquet   (clean, deduped, language-tagged)
       └▶ output/keyword_translations.json          (multilingual keyword dictionary)
            └▶ output/scored_data/year=YYYY/*.parquet   (every ad + genai_flag + matched keywords)
                 └▶ output/aggregation/*.parquet         (firm / time / industry / occupation tables)
```

## Step 0 — `config.yaml` (the control panel)

[`pipeline/config.yaml`](../pipeline/config.yaml) is the single place every tunable parameter
lives. Nothing in the pipeline is hard-coded that belongs here. The important groups:

- **Data access & column mapping** — `data_path`, and a `column_mapping` block that maps the
  pipeline's internal abstract names (`date`, `content`, `job_id`, the nested `company`
  struct fields, …) to the *actual* column names in the x28 dataset. This abstraction is why
  the pipeline can be pointed at a different dataset by editing only the mapping.
- **Phase I filters** — `country_filter` (CH), `exclude_recruiters`, `exclude_micro_enterprises`,
  `allowed_languages` (de/fr/it/rm/en), and the per-size-class sanity-check alphas
  (`sanity_alpha_sme/medium/large`).
- **Phase II validation** — `pre_period_end_year` (2021), `validation_year` (2025), the DHS and
  ratio thresholds that split keywords into Groups 1–4, and the flag-rule knobs
  (`min_g1_required`, `min_g2_for_rule2`, `min_g2g3_for_rule3`, `enforce_occupation_filter`,
  `min_occupation_intensity`).
- **Phase III metrics** — `adoption_threshold_ads`, `min_postings_for_intensity`.

It is loaded and type-checked into a `PipelineConfig` dataclass by
[`pipeline/config_loader.py`](../pipeline/config_loader.py). Every downstream function receives
this `config` object and reads its parameters from there.

## Step 1 — `main.py` (the orchestrator)

[`main()`](../pipeline/main.py) does four things:

1. Parses `--config` and `--skip-to`. `--skip-to` lets you resume from any stage
   (`phase1`/`lang`/`translate`/`phase2`/`phase3`).
2. Sets up colored/plain logging.
3. Loads the config.
4. Calls the six stages in order, each guarded by `--skip-to` and by `_ensure_dir(...)`,
   which **exits with an error if the previous stage's output folder is missing or empty.**
   That guard is the explicit statement of "stage N depends on stage N-1's output."

Everything is passed between stages through `output_dir` on disk — the only in-memory handoff
is the `translations` dict, which `translate_keywords` returns and `score_postings` consumes.

## Step 2 — `data_preparation.py` → Phase I

[`prepare_data(config)`](../pipeline/data_preparation.py) turns the raw `Data/*.parquet` files
into clean, deduplicated, per-year files. Four numbered steps:

1. **Filter & shard.** Each raw file is filtered (country/date/recruiter/micro-enterprise) and
   its nested `company` struct is flattened into flat columns (`_firm_id`, `_size_id`,
   `_industry`, …). Rows are split *by year* and written as one shard per source file:
   `prepared_data/year=YYYY/<sourcefile>.parquet`.
2. **Consolidate & dedup.** Per year, all shards are concatenated, **deduplicated on the
   `job_id` key**, written to a single `part.parquet`, and the shards are deleted.
3. **Sanity check.** A log-normal upper confidence bound (per year × firm-size class) flags
   firms posting an implausibly high number of ads.
4. **Remove outliers.** Those flagged firm-years are anti-joined out of each `part.parquet`.

**Output:** `output/prepared_data/year=YYYY/part.parquet` — one clean file per year.

## Step 3 — `language_detection.py`

[`detect_languages(config)`](../pipeline/language_detection.py) loads a FastText model
(`lid.176.ftz`, auto-downloaded if missing) and adds a **`detected_language`** column to each
year's `part.parquet`, writing back in place (idempotent — skips a file that already has the
column). Then `_restrict_to_allowed_languages` drops any posting whose language is not in
`config.allowed_languages`.

## Step 4 — `keyword_translation.py`

Back in `main.py`, the pipeline scans the prepared data for languages with **>100 postings**,
then calls [`translate_keywords(detected_languages, config)`](../pipeline/keyword_translation.py).
The keyword source is three Python files in [`../keyword_lists/`](../keyword_lists/) (Stanford
core / Hosseini mid / broad candidates), deduplicated across layers. Brand names
(`ChatGPT`, `Claude`, `LLM`, …) are kept verbatim as "proper nouns"; everything else is
translated into each detected language via Google Translate. The result is cached to
`output/keyword_translations.json`, keyed by a **hash of the keyword files**, so it only
re-translates when the lists change. Returns the `translations` dict.

## Step 5 — `keyword_scoring.py` → Phase II

[`score_postings(translations, config)`](../pipeline/keyword_scoring.py) is the heart of the
pipeline. It runs in four passes, each checkpointed to disk so re-runs skip completed work:

- **Pass 1** — scan every year, count keyword hits into `{(year, month, lang, keyword): count}`.
  No ad-level data held in RAM. → `pass1_checkpoint.json`
- **Pass 2** — for each keyword, fit an OLS trend on the 2012→2021 baseline, predict 2025,
  and assign **Group 1/2/3/4** from the DHS + ratio thresholds. → `keyword_group_map.json`,
  `validation_audit.json`
- **Pass 2.5** — re-scan **2023+** to measure keyword co-occurrence and which **occupations**
  concentrate the Group-1 signal. → `g1_dominant_occupations.csv`,
  `occupation_intensity_stats.json`
- **Pass 3** — score every ad with the flag rules, writing the full enriched rows.
  → `output/scored_data/year=YYYY/*.parquet`

## Step 6 — `aggregation.py` → Phase III

[`aggregate_results(config)`](../pipeline/aggregation.py) streams over `scored_data` and rolls
the flagged ads up into eight parquet tables in `output/aggregation/`: firm-level, overall
time-series, an adoption index (cumulative first-adopter share per year), top-20 intensity
firms, industry, industry-by-year, occupation, and keyword time-series. No matching happens here — it is all
`group_by(...).agg(genai_flag.sum())` reductions of what Phase II produced.

---

# Part 2 — Detailed Deep Dive

This part explains *how* each stage works internally. Where the logic is subtle (the
sanity-check statistics, the regex machinery, the regression/group math, the flag rules), it
is spelled out in full.

## 2.0 — Config loading (`config_loader.py`)

`load_config()` reads the YAML, then constructs two dataclasses:

- `ColumnMapping` — the abstract-name → dataset-column map.
- `PipelineConfig` — every typed parameter, including a `col: ColumnMapping` field.

Each `raw.get("key", default)` supplies a fallback if a key is missing. **These fallback
defaults are kept in sync with the shipped `config.yaml`** (e.g. `min_g2g3_for_rule3 = 3`,
`min_occupation_intensity = 0.02`), so an incomplete config behaves the same as the shipped
one. The defaults only fire when a key is absent; the live YAML always wins.

## 2.1 — Phase I: Data Preparation (`data_preparation.py`)

### `_flatten_and_filter(f_path, config)`
Loads one raw parquet lazily and:
- Parses the date column (`tst_created`) from string to datetime.
- Pulls fields out of the nested `company` struct into flat columns: `_firm_id`,
  `_company_name`, `_is_recruiter`, and (if present) `_size_id`, `_size_name`, and the first
  `INDUSTRY` entry from `company.metadata` → `_industry`. Missing sub-structs fall back to
  `"unknown"`/`null` so the schema is stable across files.
- Calls `_apply_basic_filters` and (optionally) drops micro-enterprises.

### `_apply_basic_filters(lf, config)`
Applies, conditionally on config: country filter (`locations` list contains `country == "CH"`),
date-range bounds, and recruiter exclusion (`_is_recruiter == False`).

### `_get_sanity_check_outliers(prepared_dir, config)` — the log-normal CI
This is the statistically interesting step. Firm posting counts are **right-skewed** so a plain mean ± SD would over-flag. Instead:

1. Count postings per `(firm, size_id, year)`.
2. Work on the `log(n+1)` scale, where the distribution is roughly normal.
3. Per `(year, size_id)` bucket, compute the mean and SD of `log(n+1)`, and an **upper** bound
   `exp(mean + z·SD) − 1`. Only the **upper tail** is flagged — a firm posting *few* ads is not
   a data-quality problem.
4. The `z` is chosen per size class from its alpha: **SME** is strict (α 0.005), **Medium**
   moderate (0.002), **Large** permissive (0.001)

Firms whose count exceeds their bucket's upper bound are returned as `(_firm_id, _year)`
outliers, which Step 4 of `prepare_data` removes.

## 2.2 — Language Detection (`language_detection.py`)

`detect_languages` applies the FastText model in batches via Polars `map_batches` with
`streaming=True`, keeping RAM flat regardless of partition size. `detect_lang_batch` strips
newlines, runs `model.predict(texts, k=1)`, and returns the top language label per ad. The
function is **idempotent**: if a `part.parquet` already has `detected_language`, it is skipped
(but its counts are still tallied for the summary log).

`_restrict_to_allowed_languages` then filters every partition to
`config.allowed_languages` (de/fr/it/rm/en). This runs once, right after detection, so every
downstream stage only ever sees permitted languages.

## 2.3 — Keyword Translation (`keyword_translation.py`)

- `_deduplicate_across_layers(l1, l2, l3)` — normalizes each keyword (lowercase, hyphens→spaces)
  and removes cross-layer duplicates with **layer-1 priority**: a term in both layer 1 and
  layer 2 stays only in layer 1. This keeps each keyword traceable to exactly one source layer.
- `_classify_keywords(keywords)` — splits each layer into *generic* terms (to be translated)
  and *proper nouns* (brand names in `NON_TRANSLATABLE_KEYWORDS`, kept verbatim across all
  languages).
- `_translate_keyword_list(keywords, target_lang)` — calls Google Translate per keyword with a
  retry/back-off (1s, 2s, 4s) and a 0.5s pause between calls; on repeated failure it falls
  back to the original English term.
- **Caching:** `_compute_keyword_hash()` hashes the three keyword-list files. The cache is
  reused only if the hash matches **and** all requested languages are already present;
  otherwise only the missing languages are translated and merged in.

The returned `translations` dict is shaped as
`translations[lang][layerN] = {"keywords": [...], "proper_nouns": [...]}`.

## 2.4 — Phase II: Scoring (`keyword_scoring.py`)

### Shared machinery (used by all passes)

**Combined regex OR-patterns** — `_build_origin_patterns` / `_build_combined`. Rather than
loop over hundreds of keywords per ad, all keyword forms for a language are fused into **one**
compiled regex joined by `|`, with each keyword wrapped in a *named capture group*
(`(?P<kw0>…)`). When the regex matches, `m.lastgroup` (read by `_fast_match_keywords`) tells you
*which* canonical English keyword matched — so a single pass over the text returns the full set
of matched keywords. Forms are sorted **longest-first** so `GPT-4` wins over `GPT`, and spaces/
hyphens are made flexible (`machine learning` == `machine-learning`).

**Dual-language matching** — every ad is scanned against the **English** combined pattern
*unconditionally* (English terms are retained as cross-language anchors). An ad is *additionally*
scanned against its **detected language's** pattern only if that language was translated
(`ad_lang != 'en' and ad_lang in local_patterns`). Consequences: English ads are matched with
English only; a language with no translation (e.g. Romansh `rm`, which Google Translate does not
support) falls back to English-only matching.

**Literal pre-filter** — before the regex runs, each batch is reduced to rows that
contain at least one keyword *literal* (`str.contains(lit, literal=True)`), skipping ~99% of
rows that cannot possibly match. The literal set (`_build_literal_prefilter_set`) is drawn from
**all** languages, so it is not English-biased.
*Known limitation:* the literal set only includes forms of **length ≥ 4 characters**, so the
four short keywords `GPT`, `LLM`, `RAG`, `T5` are excluded from the pre-filter. An ad whose
*only* signal is one of those bare tokens (with no longer companion term like `GPT-4` or
`ChatGPT`) is dropped before regex and never matched. In practice these short tokens almost
always co-occur with a ≥4-char term, so the recall loss is minimal. (The `<4` guard exists
because the pre-filter does *substring* matching, and a short token like `rag` would match
common German words — `Vertrag`, `Auftrag`, `Beitrag`, `tragen` — defeating the filter's
purpose. The word-boundary regex used for the *actual* match would not false-positive on those,
but the substring pre-filter would pass nearly every German ad.)

### Pass 1 — count collection (`_collect_metrics`, `_year_scan_worker`, `_metrics_scan_worker`)
Scans every year (in parallel across up to 3 worker processes) and accumulates four
counters, holding **no ad-level data** in memory:
- `counts`   — `{(year, month, lang, keyword): hits}`
- `totals`   — `{(year, month, lang): ad count}`
- `global`   — `{(year, month): ad count}`
- `occ_2025` — `{occupation: ad count}` for the validation year (the `_2025` name tracks
  `config.validation_year`, not literally 2025).

Because compiled regex can't be pickled to worker processes, each worker rebuilds the patterns
from `translations` internally; the large `translations` dict is written to a temp JSON file
and loaded per-worker rather than pickled (avoids an N×RAM spike). Result is cached to
`pass1_checkpoint.json` (tuple keys serialized as pipe-joined strings).

### Pass 2 — regression & group assignment (`_perform_regression_validation`)
For each keyword × language, using the Pass-1 monthly counts:

1. **Baseline series.** Sum hits per year for the pre-period **2012→2021**. `x = year − 2012`,
   `y = hits`.
2. **Fit a trend.** `LinearRegression(x, y)` is plain OLS — it computes the historical slope
   and intercept (2012–2021 growth) and **predicts** what 2025 should have been if that trend
   simply continued: `y_pred = predict(2025 − 2012)`.
3. **Compare prediction to reality.** Actual 2025 hits = `y_hits_2025`. Two statistics:
   - **DHS** = `2·(actual − predicted) / (actual + predicted)` — a symmetric percent
     difference bounded in **[−2, +2]**. ≈0 means "on trend"; near +2 means "2025 massively
     exceeds what the trend predicted" — a structural break. The symmetric form avoids blowups
     when `predicted ≈ 0`, which is exactly the regime emerging keywords live in.
   - **ratio** = `2025 hits / average(2012–2018 hits)`; `div_by_zero` marks a keyword with a
     literally zero pre-period baseline.
4. **Assign the group** (thresholds from config):

   | Group | Condition | Meaning |
   |-------|-----------|---------|
   | **G1** | zero baseline **AND** DHS ≥ 1.95 | brand-new explosive term (e.g. ChatGPT) — strongest signal |
   | **G2** | DHS ≥ 1.72 **AND** (zero baseline OR ratio > 11.8) | strong surge, existed somewhat before |
   | **G3** | DHS ≥ 1.00 **AND** (zero baseline OR ratio > 20.0) | weak "helper" — only counts in combination |
   | **G4** | everything else | recorded, contributes to no flag |

5. **Minimum-signal floor.** A keyword in G1/G2 with **fewer than 3** total 2025 hits across all
   languages is demoted to G3 — a 1–2 occurrence "spike" is noise, not a trend. (The `3` is
   hard-coded in this function.)
6. **English-dominant rule.** Groups are assigned per language, but English is the authority:
   - English = **G4** → force **all** languages to G4 (no signal confirmed in the source
     language; a noisy translated spike can't promote it).
   - English = **G3** → cap **all** languages at G3.
   - English = **G1/G2** → each language keeps its own DHS-assigned group.

   `overall_groups[kw]` is set to the **English** group, which Pass 2.5 and Pass 3 use.

Outputs `keyword_group_map.json` and a full per-keyword `validation_audit.json` (every DHS,
slope, yearly-hit series). The Group-3 keywords *are* the "helper" set — there is no separate
helper list, since G3 is defined exactly as the helper criterion (DHS ≥ 1.0 and ratio > 20).

### Pass 2.5 — co-occurrence & occupation intensity (`_compute_cooccurrence_and_intensity`)
Re-scans **2023+ only** (the post-ChatGPT period). For every keyword it measures how often it
appears in an ad *alongside* a G1 / G2 / G3 keyword (a "partnership" rate), using vectorized
Polars list operations. Separately, for the validation year it counts, per **occupation**, how
many ads contain a G1 keyword, and divides by the occupation's total ads to get a G1
**intensity**. It then selects the set of **AI-intensive occupations**: sorted by intensity
descending, accumulate occupations until 80% of all G1 signal is covered
(`g1_occupation_coverage_threshold`), with a floor that every included occupation must have
intensity ≥ `min_occupation_intensity` (2%). Outputs `g1_dominant_occupations.csv` and
`occupation_intensity_stats.json`.

### Pass 3 — scoring every ad (`_score_all_postings`, `_score_batch_worker`, `_year_score_worker`)
For each ad: match keywords (dual-language as above), bucket each match into `m_g1` / `m_g2` /
`m_g3` by its `group_by_lang` group (a match counts toward `m_g3` exactly when its group is 3 —
the Group-3 keywords *are* the helper set by definition, so no separate membership check is
needed). Then four **independent** rules, OR-ed into the final `genai_flag`:

- **Rule 1** — `len(m_g1) ≥ min_g1_required` (=1). A single G1 keyword flags the ad. Allowed to
  be this aggressive because a term only *becomes* G1 if it had a zero pre-2022 baseline and a
  DHS ≥ 1.95 — a genuinely new, diagnostic term.
- **Rule 2** — `≥ min_g2_for_rule2` G2 keywords **AND** (if `enforce_occupation_filter`) the
  ad's occupation is in the AI-intensive set from Pass 2.5. G2 terms are strong but slightly
  more ambiguous than G1, so they require occupational corroboration.
- **Rule 3** — `(len(m_g2) + len(m_g3)) ≥ min_g2g3_for_rule3` (=3) **AND** `len(m_g2) ≥ 1`. A
  *cluster* rule: several weak/medium signals together, but at least one must be a real G2.
- **Rule C (Copilot)** — "Copilot" is ambiguous (Microsoft Copilot vs. aircraft co-pilot). It
  only fires if "Copilot" matched **and** "Microsoft"/"GitHub" appears in the text. An
  **aviation exclusion** also strips "copilot" from the G2 set when the occupation is
  Pilot/Kopilot/Co-Pilot, so flight jobs are not falsely flagged.

The design is a graduated trust model: the more certain the signal (G1), the less
corroboration required; the weaker the signal (G3), the more it must combine with others.

Each scored ad gets `genai_flag`, the per-rule flags, group-match booleans, counts, and the
matched-keyword lists, written to `output/scored_data/year=YYYY/*.parquet`.

> **Implementation note:** `_score_batch_worker` (sequential) and `_year_score_worker`
> (parallel) are **twins** — they implement the identical flag logic. If you change a scoring
> rule, change it in **both**.

### Checkpointing
`score_postings` skips work that already exists: if `pass1_checkpoint.json` is present it loads
the counts; if `keyword_group_map.json` + `g1_dominant_occupations.csv`
all exist it skips straight to Pass 3. Delete those files to force a re-run of the relevant
pass.

## 2.5 — Phase III: Aggregation (`aggregation.py`)

`aggregate_results` uses Polars `scan_parquet` + `.collect(streaming=True)` (the out-of-core
engine) so it can aggregate ~20M rows on disk. It produces eight tables:

- **`firm_level`** — per firm: total postings, GenAI postings, group-1/2 counts, GenAI rate.
- **`time_series`** — per year: total vs GenAI postings and percentage.
- **`adoption_index`** — the extensive-margin measure. A firm is an "adopter" in a year if its
  GenAI-flag sum ≥ `adoption_threshold_ads`; the table tracks each firm's **first** adoption
  year, the cumulative count of adopters, and that count as a **share of all firms posting**
  that year.
- **`firm_intensity_top20`** — per year, the top-20 firms by GenAI intensity (share of their
  own ads flagged), filtered to firms with ≥ `min_postings_for_intensity` ads to avoid
  1/1 = 100% outliers.
- **`industry_level`** — per `_industry` (the field extracted in Phase I), GenAI percentage.
- **`industry_time_series`** — per `_industry` × year, GenAI percentage — the table for *"does industry X have a relatively high frequency of AI ads at a given time"*.
- **`occupation_level`** — per occupation × year, GenAI percentage.
- **`keyword_time_series`** — explodes `matched_keywords` and counts, per keyword × year, how
  many ads matched and how many were flagged, broken down by which rule did the flagging.

All eight are written to `output/aggregation/*.parquet`.

---

## Stage → output map (quick reference)

| Stage | Function | Reads | Writes |
|-------|----------|-------|--------|
| Phase I | `prepare_data` | `Data/*.parquet` | `output/prepared_data/year=*/part.parquet` |
| Language | `detect_languages` | `prepared_data/` | same files + `detected_language` column |
| Translation | `translate_keywords` | `keyword_lists/*.py`, prepared data (lang counts) | `output/keyword_translations.json` |
| Phase II | `score_postings` | `prepared_data/`, `translations` | `output/scored_data/year=*/`, several `*.json`/`*.csv` checkpoints |
| Phase III | `aggregate_results` | `scored_data/` | `output/aggregation/*.parquet` |

When a comment and the code disagree, **trust the code and the live `config.yaml`** — they are
the source of truth this document was written against.
