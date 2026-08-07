# GenAI Adoption Pipeline

Measuring **Generative-AI adoption in the Swiss labor market** from x28 job-posting
data, via keyword-based classification against a validated keyword list (Domenico
Job's Bachelor's thesis, University of Zurich). Pipeline and analysis scripts live
in this folder; see
[`genai-adoption-pipeline/README.md`](genai-adoption-pipeline/README.md) for how to
actually run it.

This is a **code-only package** — no data, no generated output. Raw x28 data isn't
included since you already have your own access to it; see "Expected data layout"
below for where the pipeline expects to find it. Results (classified data,
dashboards) are being sent separately.

## Repo structure

```
genai-adoption-pipeline/     # the pipeline itself - see its own README
├── pipeline/                # live modules (data prep, language mapping, classification)
├── keyword_lists/           # master_keywords.json - the active, finalized (117-keyword) list
├── Explanations/            # methodology docs (Thesis_Overview_README.md now flagged - see below)
└── requirements.txt
code/
└── mine/                    # orchestration, export, and analysis/dashboard scripts
```

## Expected data layout

Not included in this package — point these at your own copy:

```
data/
├── raw/          # x28 DuckDB dumps + ISCO/AVAM reference tables
└── processed/    # Parquet exports (regenerable from data/raw/ via
                   # code/mine/export_duckdb_to_parquet.py)
```

`genai-adoption-pipeline/pipeline/config.yaml`'s `data_path` currently points at
a machine-specific absolute path — update it to your own `data/processed/x28_parquet/`
location before running.

## Current methodology (v7)

**This replaced the original DHS-regression / structural-break approach** described
in `Explanations/Thesis_Overview_README.md` — that document is now flagged
accordingly (see its banner). The active methodology, implemented in
`code/mine/v7_scoring.py`, agreed with you on 2026-07-17 and refined 2026-08-03/06:

- For each keyword, per language: `yhat` = OLS trend fit on 2016-2020 annual
  counts, predicted at 2025; `mean` = average of the same 2016-2020 counts;
  benchmark `B = max(yhat, mean)` (never negative).
- **G1**: zero matches in 2016-2020 (all languages combined) AND at least
  one match in the 2023-2025 validation window (all languages combined).
- **G2**: 2025 count ≥ `c * B` AND ≥ 10, within a single language — then
  propagates to every language form of that keyword once true in any one
  language.
- **Ad-level flag**: ≥1 G1 keyword match, OR ≥ `k` G2 keyword matches, OR
  Rule C (Copilot + Microsoft/GitHub in the same ad — currently a structural
  no-op, since bare "Copilot" was removed from the keyword list on 2026-07-30
  for false-positive collisions with aviation co-pilot ads; genuine Copilot
  mentions still get flagged via the unambiguous "GitHub Copilot"/"Microsoft
  Copilot" keyword forms).
- `c` and `k` are swept over grids (`C_GRID = [1.5, 3, 5, 10]`,
  `K_GRID = [1, 2, 3, 4, 5]` in `v7_scoring.py`) rather than fixed — **not
  yet calibrated against a hand-labeled sample**, flagged as an open item in
  `v7_scoring.py`'s own docstring, not silently decided.

`pipeline/config.yaml` still contains the old DHS/ratio thresholds — they're
explicitly marked `NOT CURRENTLY USED` in the file itself, kept only in case a
future dataset has enough pre-period history to bring that approach back.

## `code/mine/` — what each script does

**Orchestration** (run in order to regenerate `prepared_data`/`classified_data`):
| Script | Purpose |
|---|---|
| `export_duckdb_to_parquet.py` | Exports an `x28_dump*.duckdb` file into per-day Parquet matching the pipeline's expected schema |
| `run_full_pipeline_step1_3.py` | **Main entry point.** Runs Phase I (data prep) → language mapping → keyword-group classification on the full dataset |
| `run_classification_only.py` | Reruns just the classification step, when prep/language steps are already done |

**v7 scoring / dashboards** (read `output/classified_data`, produce the results):
| Script | Purpose |
|---|---|
| `v7_scoring.py` | The current scoring methodology (see above) — also runnable standalone for the grid-sweep CSVs |
| `build_v7_dashboard.py` | v7 methodology dashboard — per-keyword G1/G2 classification, breakeven-`c` table, `c`/`k` grid sweep charts |
| `build_results_dashboard_v4.py` | General results dashboard — classification breakdown, adoption trends, occupation/industry/firm-size views |
| `build_layer_breakdown_dashboard.py` | Corroboration/QA dashboard — keyword coverage by source layer (Stanford AI Index / Hosseini & Lichtinger / LLM-brainstormed) |

**QA / supporting analysis**:
| Script | Purpose |
|---|---|
| `classification_sanity_check.py` | Prints sample matched ads per group for manual spot-checking |
| `occupation_genai_share_plot.py` | Occupation-level GenAI-share sanity plot (x28 → AVAM → ISCO join) |

## `genai-adoption-pipeline/`

See [its own README](genai-adoption-pipeline/README.md) for the pipeline's internal
structure, how to run it, and what lands in `output/`.

---
*Author: Domenico Job — Bachelor's thesis, University of Zurich.*
