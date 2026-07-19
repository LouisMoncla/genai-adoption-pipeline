# GenAI Adoption Pipeline

**Author:** Domenico Job  
**Thesis:** *The Adoption of Generative AI in the Swiss Labor Market: An Empirical Firm-Level Analysis*  
**Supervisor:** Prof. David Dorn, University of Zurich  

---

## What This Pipeline Does

This pipeline measures **firm-level adoption of Generative AI** from job posting data. It processes a large collection of job advertisements (Parquet format), identifies which postings contain Generative AI-related terminology using an econometrically validated keyword classification system, and aggregates the results at the firm, occupation, industry, and time-series level.

The keyword validation is data-driven: keywords are only retained if they show a statistically significant structural break in posting frequency after November 2022 (the release of ChatGPT), measured using the DHS growth rate (Davis, Haltiwanger & Schuh 1995).

---

## Folder Structure

```
genai-pipeline-handover/
├── requirements.txt               # Python dependencies
├── schema_ads.json                # Expected Parquet schema for the input data
│
├── pipeline/                      # Core modules + config (edit config.yaml only)
│   ├── config.yaml                # All parameters to adjust for a new dataset
│   ├── main.py                    # Entry point — run with: python -m pipeline.main
│   ├── config_loader.py           # Config parsing
│   ├── data_preparation.py        # Phase I: filtering, dedup, sanity checks
│   ├── language_detection.py      # FastText language identification
│   ├── keyword_translation.py     # Multi-language keyword translation
│   ├── keyword_scoring.py         # Phase II: econometric validation + flagging
│   └── aggregation.py             # Phase III: firm-level aggregation
│
├── keyword_lists/                 # Keyword definitions (grouped by source)
│   ├── layer1_stanford.py         # G1 anchor terms (Stanford AI Index 2025)
│   ├── layer2_hosseini.py         # G2 helper terms (Hosseini & Lichtinger 2025)
│   ├── layer3.py                  # G3 broad candidates (~97 terms)
│   └── translated_keywords/       # Swiss-language translations of the keywords
│       ├── keyword_translations_swiss.xlsx   # Readable EN/DE/FR/IT table (for review)
│       └── keyword_translations.json         # Same data, machine-readable
│
├── Explanations/                  # Documentation for the receiver
│   ├── Pipeline_Code_Explanation.md   # Detailed walkthrough of the raw code (start here)
│   ├── Thesis_Overview_README.md       # This file — high-level overview & usage
│   ├── CONFIG_TUNING_CHEATSHEET.md      # How to make detection stricter / more lenient
│   └── Declaration_of_AI_Usage.md       # Disclosure of AI-assisted development
│
└── output/                        # Generated artefacts (ships with the translation cache)
    ├── keyword_translations.json  # Pre-computed EN/DE/FR/IT(+others) keyword translations
    ├── prepared_data/             # Phase I output (year-partitioned Parquet)
    ├── scored_data/               # Phase II output (with genai_flag column)
    └── aggregation/               # Phase III tables (firm_level, time_series, …)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Adapting for a New Dataset

All dataset-specific parameters are controlled by `pipeline/config.yaml`. Open it and adjust the following:

### 1 — Point to your data
```yaml
data_path: "Data/"   # Change to the folder containing your Parquet files
```

### 2 — Map your column names
The pipeline uses an internal abstraction layer. Map your dataset's column names here:
```yaml
column_mapping:
  date:                 "your_date_column"        # Posting creation date
  content:              "your_text_column"         # Full job description text
  locations:            "your_locations_column"    # Location struct/array
  job_id:               "your_dedup_key"           # Deduplication key
  company_struct:       "company"                  # Struct containing firm info
  company_id:           "id"                       # Firm identifier (inside struct)
  company_name:         "name"                     # Firm name (inside struct)
  company_is_recruiter: "is_recruiter"             # Boolean recruiter flag
  company_metadata:     "metadata"                 # Metadata struct
  company_size_struct:  "size"                     # Size struct (inside company)
  company_size_id:      "id"                       # Size code (inside size struct)
  company_size_name:    "name"                     # Size label (inside size struct)
```

### 3 — Set the date range
```yaml
date_range_start: "2012-01-01"   # null = no lower bound
date_range_end:   null           # null = no upper bound
```

### 4 — Set the country filter
```yaml
country_filter: "CH"   # ISO 3166-1 alpha-2 code, or null to disable
```

### 5 — Adjust size class codes if needed
The pipeline filters out micro-enterprises by size class ID:
```yaml
micro_enterprise_id: "57000001"  # Change to match your dataset's size coding
```
If your dataset does not have size classes, set `exclude_micro_enterprises: false`.

---

## Running the Pipeline

Run from the **repository root** (the folder containing `pipeline/`), using module syntax so the `pipeline` package resolves:

```bash
# Full run (all phases)
python -m pipeline.main

# Use a custom config file
python -m pipeline.main --config my_config.yaml

# Skip to a later phase (if earlier phases already completed)
python -m pipeline.main --skip-to phase2   # skip Phase I, start from keyword scoring
python -m pipeline.main --skip-to phase3   # skip to aggregation only
```

Available skip targets: `phase1`, `lang`, `translate`, `phase2`, `phase3`

> **Note:** always launch from the repo root with `python -m pipeline.main`
> (not `python pipeline/main.py`) so the `pipeline` package and the default
> `pipeline/config.yaml` path resolve correctly.

---

## Pipeline Phases

| Phase | What it does | Output |
|---|---|---|
| **Phase I** | Loads Parquet files, filters by date/country/size, deduplicates, runs statistical sanity checks on firm posting volumes | `output/prepared_data/year=YYYY/*.parquet` |
| **Language** | Detects language of each posting via FastText (DE/FR/IT/EN/other) | Language column added to prepared data |
| **Translate** | Loads the pre-computed keyword translations from `output/keyword_translations.json` (shipped with this package). Only contacts Google Translate if a detected language is missing from that cache. | `output/keyword_translations.json` |
| **Phase II** | Runs OLS regression per keyword to validate disruption (DHS growth rate); assigns keywords to Groups 1–4; flags postings using Triple-Lock rules | `output/scored_data/year=YYYY/*.parquet` |
| **Phase III** | Aggregates flagged postings by firm×year, industry, **industry×year**, occupation, and time series | `output/aggregation/` |

---

## Output Structure

After a successful run, `output/` contains:

```
output/
├── prepared_data/        # Phase I output (year-partitioned Parquet)
├── scored_data/          # Phase II output (with genai_flag column)
├── aggregation/          # Phase III tables (firm_level, time_series, etc.)
├── keyword_translations.json    # Cached keyword translations
├── keyword_group_map.json       # Keyword→Group assignments (incl. group_by_lang)
├── validation_audit.json        # Per-keyword DHS / regression stats
└── g1_dominant_occupations.csv  # AI-intensive occupation set (Pass 2.5)
```

---

## Key Methodological Parameters (config.yaml)

| Parameter | Default | Meaning |
|---|---|---|
| `pre_period_end_year` | 2021 | OLS baseline estimated on data up to this year |
| `validation_year` | 2025 | Year used to measure the post-ChatGPT shock |
| `dhs_group1_threshold` | 1.95 | Minimum DHS growth rate for a keyword to reach Group 1 |
| `dhs_group2_threshold` | 1.72 | Minimum DHS growth rate for Group 2 |
| `dhs_group3_threshold` | 1.00 | Minimum DHS growth rate for Group 3 |
| `ratio_group2_threshold` | 11.8 | Minimum Average Ratio (2025 vs 2012–2018 mean) for Group 2 |
| `ratio_group3_threshold` | 20.0 | Minimum Average Ratio for Group 3 |
| `min_g1_required` | 1 | Rule 1: ≥1 Group 1 keyword → flag |
| `min_g2_for_rule2` | 1 | Rule 2: ≥1 Group 2 keyword + AI-intensive occupation → flag |
| `min_g2g3_for_rule3` | 3 | Rule 3: ≥3 combined G2+G3 keywords (≥1 must be G2) → flag |

---

*Pipeline version 13 — April 2026*
