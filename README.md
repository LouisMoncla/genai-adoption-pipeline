# GenAI Adoption Pipeline

Measuring **Generative-AI adoption in the Swiss labor market** from x28 job-posting
data, via keyword-based classification against a validated keyword list (Domenico
Job's Bachelor's thesis, University of Zurich). Pipeline, analysis scripts, and
supporting data live in this repo; see
[`genai-adoption-pipeline/README.md`](genai-adoption-pipeline/README.md) for how to
actually run it.

## Repo structure

```
Internship/
├── genai-adoption-pipeline/   # the pipeline itself - see its own README
│   ├── pipeline/              # live modules (data prep, language detection, classification)
│   │   └── legacy/            # superseded Phase II modules, kept for provenance - not run
│   ├── keyword_lists/         # master_keywords.json (active) + legacy/ raw candidate lists
│   ├── Explanations/          # deep-dive docs (some marked LEGACY where superseded)
│   ├── output/                # generated artefacts (gitignored except the translation cache)
│   └── venv/                  # local virtualenv (gitignored)
├── code/
│   ├── mine/                  # orchestration, export, and analysis/dashboard scripts
│   └── from_jeremias/         # R scripts (need KOF server DB access to run)
└── data/
    ├── raw/                   # x28 DuckDB dumps + reference tables (gitignored, proprietary)
    └── processed/             # exported Parquet + scratch DBs (gitignored, regenerable)
```

`notes/` (internal working notes/meeting prep) and `outputs/` exist locally but are
gitignored — not part of this repo.

## `code/mine/` — what each script does

**Orchestration** (run these, in order, to regenerate everything):
| Script | Purpose |
|---|---|
| `export_duckdb_to_parquet.py` | Exports an `x28_dump*.duckdb` file into per-day Parquet matching the pipeline's expected schema |
| `build_master_keywords.py` | Merges `data/raw/validated_keywords.json` (Domenico's Table 4) with translated forms into `keyword_lists/master_keywords.json` |
| `run_full_pipeline_step1_3.py` | **Main entry point.** Runs Phase I (data prep) → language mapping → keyword-group classification on the full dataset |
| `run_classification_only.py` | Reruns just the classification step, when prep/language steps are already done |

**Analysis / reporting** (read `output/classified_data`, produce plots/tables):
| Script | Purpose |
|---|---|
| `build_results_dashboard_v3.py` | **Start here for results.** Builds the single-file HTML dashboard — see `genai-adoption-pipeline/README.md`'s Output section for what's in it |
| `keyword_match_counts_by_group_layer.py` | How many of the 49 validated keywords produced ≥1 match, by Group and by source layer |
| `group_strategy_analysis.py` | Tests Group 1/2/3 combination strategies (Option A/B/C) and picks the corroboration threshold `k` |
| `occupation_genai_share_plot.py` | Occupation-level GenAI-share sanity plot (x28 → AVAM → ISCO join) |
| `plot_postings_volume.py` | Quick posting-volume plots straight from exported Parquet |
| `classification_sanity_check.py` | Prints sample matched ads per group for manual spot-checking |

`code/from_jeremias/` holds two R scripts (`test_ads_isco3_timeseries.R`,
`wfh_analysis.R`) written by the collaborator on the KOF side — kept as reference,
not runnable here (they need direct KOF Postgres access).

## `data/`

- `data/raw/` — the original x28 DuckDB dumps (large, proprietary, never modified)
  plus small reference tables (ISCO/AVAM mappings) and `validated_keywords.json`
  (the one raw-data file that IS tracked in git — small, hand-verified, needed for
  traceability).
- `data/processed/` — Parquet exports and DuckDB scratch/staging files, fully
  regenerable from `data/raw/` via `export_duckdb_to_parquet.py`. Gitignored
  entirely.

## `genai-adoption-pipeline/`

See [its own README](genai-adoption-pipeline/README.md) for the pipeline's internal
structure, how to run it, and a full reference of what's in `output/`.

---
*Author: Domenico Job — Bachelor's thesis, University of Zurich.*
