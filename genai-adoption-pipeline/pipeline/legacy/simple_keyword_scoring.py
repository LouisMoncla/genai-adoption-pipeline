"""
Simplified ad-flagging step — replaces keyword_scoring.py's DHS-regression
Phase II for this dataset.

WHY THIS EXISTS: keyword_scoring.py's regression (score_postings) needs a
2012-2021 pre-ChatGPT baseline to detect which keywords show a statistically
significant post-ChatGPT jump. The x28 dataset only covers Dec 2022 - Nov
2025 - there's no pre-period to regress against, so that approach can't run
on this data. Decided with Jeremias 2026-07-17: skip the regression and the
G1/G2/G3/G4 tiering it produces entirely. Every keyword in layer1_stanford.py
/ layer2_hosseini.py / layer3.py is treated as an equally-weighted signal —
no keyword is "stronger" than another here.

WHAT IT PRODUCES, per ad:
  - genai_flag  : True if the ad's text matches >=1 keyword from any layer.
  - genai_score : count of DISTINCT keywords matched (repeated mentions of
                  the same keyword, e.g. "ChatGPT" three times in one ad,
                  still count once). This is the simplest possible starting
                  definition for a continuous score.

    Jeremias hasn't picked a final formula yet — he mentioned "count of
    matched keywords" or "share of matched keywords relative to total
    keywords checked" as options. This uses the count. Getting the share
    instead is a one-line change: genai_score / TOTAL_KEYWORD_COUNT (the
    total candidate pool size is logged below, and is len(all_kws) in this
    file if needed). Not adding it as a second column yet, since Jeremias
    said not to build out logic beyond what's needed for a first run.

WHAT'S REUSED FROM keyword_scoring.py: the keyword-matching machinery itself
(regex compilation from translations, the literal pre-filter, the "which
keywords appear in this text" matcher) is NOT regression-specific — it's the
same "does this text contain this keyword" question regardless of how the
result gets used afterward. Reusing it here instead of rewriting it avoids
duplicating tested, working code.

WHAT'S DIFFERENT FROM keyword_scoring.py's Pass 3, on purpose:
  - No multiprocessing. Pass 3 scores years in parallel worker processes;
    this runs sequentially, one year at a time. Simpler code, and this
    machine only has 7.7 GB RAM — running multiple Python processes that
    each hold a copy of the translations dict is a real risk here, not
    worth it for a first run.
  - No G1/G2/G3 group lookup, no occupation gate, no Copilot special case.
    Just: matched anything -> flagged, count what matched -> score.

WHAT'S KEPT FOR COMPATIBILITY: aggregation.py (Phase III) reads several
columns this module doesn't have a real use for — flag_rule_1/2/3/copilot
(which rule fired, for the keyword_time_series breakdown) and
group1_match/group2_match (for firm_level's group1_count/group2_count).
Rather than editing aggregation.py too, this writes those same column names
with simplified values (flag_rule_1 = genai_flag, everything else = False)
so Phase III runs unmodified. They carry no independent meaning under this
scheme — genai_flag and genai_score are the two columns that actually matter.
"""

import logging
from pathlib import Path

import polars as pl

from pipeline.config_loader import PipelineConfig
from pipeline.keyword_scoring import (
    _collect_all_keywords,
    _build_origin_patterns,
    _build_literal_prefilter_set,
    _fast_match_keywords,
)

logger = logging.getLogger(__name__)


def score_postings_simple(translations: dict, config: PipelineConfig) -> None:
    """Entry point — called from main.py instead of keyword_scoring.score_postings()."""
    output_dir = Path(config.output_dir)
    data_root = output_dir / "prepared_data"
    scored_root = output_dir / "scored_data"
    scored_root.mkdir(parents=True, exist_ok=True)

    logger.info("=== Simplified Scoring: flag + score every ad against layer1/2/3 keywords ===")
    logger.info("    (DHS regression skipped - no pre-2022 baseline data available)")

    all_kws = _collect_all_keywords(translations)
    en_patterns, local_patterns = _build_origin_patterns(translations, all_kws)
    literals = _build_literal_prefilter_set(translations, all_kws)
    logger.info(f"  {len(all_kws)} distinct keywords loaded across layer1/2/3 (all languages combined).")

    lit_exprs = [pl.col("_cl").str.contains(lit, literal=True) for lit in sorted(literals)]

    for p_dir in sorted(data_root.glob("year=*")):
        year_name = p_dir.name
        part_file = p_dir / "part.parquet"
        files = [part_file] if part_file.exists() else sorted(p_dir.glob("*.parquet"))
        if not files:
            continue

        year_out = scored_root / year_name
        year_out.mkdir(exist_ok=True)

        for pf in files:
            df = pl.read_parquet(pf)
            if df.is_empty():
                continue

            # Same trick keyword_scoring.py uses: skip the expensive per-row
            # regex loop for rows that can't possibly contain any keyword.
            pre_mask = (
                df.select(
                    pl.col(config.col.content).str.to_lowercase().fill_null("").alias("_cl")
                )
                .with_columns(pl.any_horizontal(*lit_exprs).fill_null(False).alias("_any"))
                ["_any"]
            )
            pre_match_set: set[int] = set(pre_mask.arg_true().to_list())

            struct_series = df.select(
                pl.struct([config.col.content, "detected_language"])
            ).to_series()

            matched_lists: list[list[str]] = []
            for i, row in enumerate(struct_series):
                if i not in pre_match_set:
                    matched_lists.append([])
                    continue

                text = row[config.col.content]
                ad_lang = row["detected_language"]

                matched = _fast_match_keywords(text, en_patterns["_combined"])
                if (
                    ad_lang != "en"
                    and ad_lang in local_patterns
                    and "_combined" in local_patterns[ad_lang]
                ):
                    matched |= _fast_match_keywords(text, local_patterns[ad_lang]["_combined"])

                matched_lists.append(sorted(matched))

            df_scored = (
                df.with_columns(
                    pl.Series("matched_keywords", matched_lists, dtype=pl.List(pl.String))
                )
                .with_columns(
                    [
                        pl.col("matched_keywords").list.len().alias("genai_score"),
                        (pl.col("matched_keywords").list.len() > 0).alias("genai_flag"),
                    ]
                )
            )

            # Compatibility stubs for aggregation.py — see module docstring.
            df_scored = df_scored.with_columns(
                [
                    pl.col("genai_flag").alias("flag_rule_1"),
                    pl.lit(False).alias("flag_rule_2"),
                    pl.lit(False).alias("flag_rule_3"),
                    pl.lit(False).alias("flag_rule_copilot"),
                    pl.col("genai_flag").alias("group1_match"),
                    pl.lit(False).alias("group2_match"),
                ]
            )

            df_scored.write_parquet(year_out / pf.name)

        logger.info(f"    {year_name}: scored.")
