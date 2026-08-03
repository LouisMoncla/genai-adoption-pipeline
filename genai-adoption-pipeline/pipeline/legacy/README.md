# Legacy pipeline modules

These five modules were the original Phase II design — a DHS-regression-based
flag/score approach (`keyword_scoring.py`) and its simplified successor
(`simple_keyword_scoring.py`), fed by `keyword_translation.py`'s live
Google-Translate-backed cache and aggregated by `aggregation.py` — all wired
together through `main.py`.

**Superseded 2026-07-17 through 2026-07-23** by `pipeline/group_classification.py`,
which matches against Domenico's actual validated keyword list
(`keyword_lists/master_keywords.json`) instead of the placeholder flag+score
approach here. The regression in `keyword_scoring.py` also needs a pre-2022
baseline this dataset doesn't have (decided with Jeremias, 2026-07-17).

Kept for provenance/history, not part of the active pipeline. The real entry
point today is `code/mine/run_full_pipeline_step1_3.py`, not `main.py`. See
the top-level `genai-adoption-pipeline/README.md` for the current pipeline.
