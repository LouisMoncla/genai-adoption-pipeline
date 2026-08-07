"""
v7 scoring methodology, per Jeremias's spec (Downloads/genai_scoring_rules_v7.xlsx,
received 2026-08-03, summarized in notes/claude_context_2026-08-03.md). Full
replacement for the old DHS+ratio group-assignment logic
(pipeline/legacy/keyword_scoring.py) - kept as a separate module rather than
overwriting the legacy one or the active pipeline/group_classification.py, per
Jeremias wanting comparisons against the old system preserved.

CORE IDEA: a single "multiple" statistic replaces the old DHS+ratio pair.
  yhat     = OLS(2016-2020 annual counts) predicted at 2025
  mean     = average(2016-2020 annual counts)
  B        = max(yhat, mean)                      (benchmark, never negative)
  G1: zero_16_20 (zero counts in ALL languages, 2016-2020) AND y2025 >= 1 (any language)
  G2: y2025 >= c*B AND y2025 >= 10, within a SINGLE language - then propagates
      to every language form of that keyword once true in any one language
  Ad flag: N_G1 >= 1  OR  N_G2 >= k  OR  Rule C (Copilot + Microsoft/GitHub)
  c and k are both TBD, calibrated via a grid sweep (this script) plus a
  hand-labelled sample (NOT done here - explicitly parked, see notes file).

KEY SIMPLIFICATION vs. the legacy Pass-1 (which scanned the full multi-
million-row corpus with multiprocessing): group_classification.py's `group`
column is "NA" if and only if ZERO of the validated keywords matched an
ad (classify_text() returns an empty dict only in that case). So every ad
with group != "NA" is the COMPLETE population of ads with >=1 keyword match
- not a sample. That was ~7,300 ads out of 6.3M with the original 49-keyword
list; ~71,400 now that master_keywords.json has grown to 117 (2026-08-06) -
still a small slice of the corpus, not a full-corpus scan. This script
re-runs full keyword matching (every keyword, not just the best-priority
group group_classification.py persisted) on just that small population -
the same approach group_strategy_analysis.py already uses - which is fast
enough to not need multiprocessing at all.

LANGUAGE ATTRIBUTION CONVENTION (matches the legacy Pass-1's own choice,
carried forward for consistency, not re-litigated by the v7 spec): a match
via a keyword's English form is counted under language bucket "en"
regardless of the ad's own detected_language (brand names appear verbatim
regardless of ad language); a match via a keyword's local-language form is
counted under the ad's own detected_language. Romansh ("rm") ads have no
distinct local forms (per the whole project's existing convention) and so
only ever contribute to the "en" bucket.

TWO JUDGMENT CALLS made here, not explicitly resolved by the spec - flagged
in the output/writeup for discussion, not silently decided:
  1. G1/G2 overlap: a keyword with a fully zero 2016-2020 pre-period has
     B=0, so y2025 >= c*0 = 0 is trivially true - meaning a G1 keyword could
     ALSO mathematically satisfy G2's multiple condition. Implemented: G1
     takes precedence, G1 keywords are excluded from the G2 set (an ad with
     a G1 keyword is flagged via N_G1 >= 1 regardless, so this only affects
     how "N_G2 keywords" is reported/counted, not final ad-flagging for
     those specific ads).
  2. Rule C is currently a no-op: bare "Copilot" was deliberately removed
     from master_keywords.json on 2026-07-30 (false-positive collisions
     with aviation co-pilot ads), so "Copilot matched" can never be true
     under today's keyword list - genuine Microsoft/GitHub Copilot mentions
     are still caught as ordinary G1-tier keywords via the unambiguous
     "GitHub Copilot"/"Microsoft Copilot" compound forms, just not via Rule
     C specifically. Implemented as literally specified (checks for a
     "Copilot" match that structurally cannot occur) rather than silently
     reinstating bare-Copilot tracking - that would be adding back
     classification logic without checking first.
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import polars as pl

from pipeline.config_loader import load_config
from pipeline.group_classification import build_keyword_patterns, load_master_keywords, normalize

TREND_START_YEAR = 2016
TREND_END_YEAR = 2020  # inclusive; 2021 explicitly excluded (GitHub Copilot
                        # preview / DALL-E / GPT-3 API all arrived that year)
VALIDATION_YEAR = 2025
G1_VALIDATION_YEARS = (2023, 2024, 2025)  # per Jeremias, 2026-08-06: G1's
    # "does it exist now" check uses this 3-year window instead of 2025
    # alone - more robust than hinging G1 status on a single year. Only
    # affects G1; G2's y2025 >= c*B condition is unchanged, still 2025 only.
MIN_VALIDATION_HITS_G2 = 10   # Fixed, within a single language
MIN_VALIDATION_HITS_G1 = 1    # marked OPEN status in the spec - not finalized

C_GRID = [0, 1.05, 1.1, 1.5, 3, 5]  # per Jeremias, 2026-08-07 - dropped 10
    # (never actually cut anything - see breakeven_c table), added 0/1.05/1.1
    # to see the low end again. c=0 is deliberately degenerate: it makes the
    # G2 rule "y2025 >= c*B" always true, so G2 collapses to just "any
    # language cleared the 10-match floor" - the benchmark-growth check
    # drops out entirely at that point. Kept for comparison, not because
    # it's a real candidate value.
K_GRID = [1, 2, 3, 4, 5]

# Keyword forms only ever exist for de/fr/it besides en (see
# build_keyword_patterns() - LANGS there is de/fr/it too). Romansh ads have
# no distinct local forms and only ever match via the "en" bucket - same
# convention as the rest of this project.
LANGS = ("en", "de", "fr", "it")


class LinearRegression:
    """Small dependency-free OLS helper - lifted from
    pipeline/legacy/keyword_scoring.py's identical class rather than
    reimplemented, since it's exactly the shape v7 needs (fit on x/y pairs,
    predict at one point)."""

    def __init__(self, x: list[float], y: list[float]):
        self.n = len(x)
        if self.n < 2:
            self.beta0, self.beta1 = 0.0, 0.0
            return
        mean_x = sum(x) / self.n
        mean_y = sum(y) / self.n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den = sum((xi - mean_x) ** 2 for xi in x)
        self.beta1 = num / den if den != 0 else 0.0
        self.beta0 = mean_y - self.beta1 * mean_x

    def predict(self, x_star: float) -> float:
        return self.beta0 + self.beta1 * x_star


def _match_ad(text: str, lang: str, en_patterns, local_patterns) -> tuple[set[str], set[str]]:
    """Returns (matched_via_en, matched_via_local) - kept separate (unlike
    group_classification.py's classify_text(), which merges them into one
    dict) because v7's per-language counting needs to know which pattern
    type each match came through."""
    norm = normalize(text or "")
    via_en = {kw for kw, _grp, pat in en_patterns if pat.search(norm)}
    via_local = {kw for kw, _grp, pat in local_patterns.get(lang, []) if pat.search(norm)}
    return via_en, via_local


def collect_matches(
    config, en_patterns, local_patterns, excluded_kws: set[str] = frozenset(),
    extra_candidate_regex: str | None = None,
):
    """Re-run full keyword matching (all 49 keywords, every group) on every
    ad with >=1 match in the existing classified_data (group != 'NA') - see
    module docstring for why this is the complete population, not a sample
    FOR THE EXISTING KEYWORD PATTERN SET.

    `extra_candidate_regex`: needed whenever the pattern set being tested
    includes a form that ISN'T in the live master_keywords.json patterns
    (e.g. the bare-LLM experiment's extra `\\bllms?\\b` pattern) - group
    != 'NA' is only a complete population with respect to patterns
    group_classification.py actually ran. An ad whose ONLY possible match
    is the new pattern would be 'NA' today and get silently missed without
    this. When set, the candidate population becomes
    (group != 'NA') OR (content_clean matches this regex, case-insensitive).

    Returns:
      counts: {(year, lang_bucket, kw): count}
      per_ad_matches: list of (ad_id, year, sorted matched-keyword list)
    """
    lf = pl.scan_parquet(Path(config.output_dir) / "classified_data" / "year=*" / "*.parquet")
    candidate_mask = pl.col("group") != "NA"
    if extra_candidate_regex:
        candidate_mask = candidate_mask | (
            pl.col("content_clean").str.to_lowercase().str.contains(extra_candidate_regex)
        )
    df = (
        lf.filter(candidate_mask)
        .select(["ad_id", "content_clean", "detected_language", "tst_created"])
        .collect(streaming=True)
    )
    print(f"  Recomputing full keyword matches for {df.height:,} candidate ads...")

    counts: Counter = Counter()
    per_ad_matches: list = []

    for ad_id, text, lang, dt in zip(
        df["ad_id"], df["content_clean"], df["detected_language"], df["tst_created"]
    ):
        year = dt.year
        via_en, via_local = _match_ad(text, lang or "en", en_patterns, local_patterns)
        via_en -= excluded_kws
        via_local -= excluded_kws
        all_matched = via_en | via_local
        if not all_matched:
            continue
        for kw in via_en:
            counts[(year, "en", kw)] += 1
        for kw in via_local:
            counts[(year, lang, kw)] += 1
        per_ad_matches.append((ad_id, year, sorted(all_matched)))

    return counts, per_ad_matches


def classify_keywords(counts: Counter, all_kws: list[str]) -> dict:
    """Per-keyword G1 status (fixed) and per-language benchmark/y2025 stats
    (used later to derive G2 status at any candidate c)."""
    pre_years = list(range(TREND_START_YEAR, TREND_END_YEAR + 1))
    x_pre = [float(y - TREND_START_YEAR) for y in pre_years]
    x_star = float(VALIDATION_YEAR - TREND_START_YEAR)

    result = {}
    for kw in sorted(all_kws):
        total_pre_all_langs = sum(counts.get((y, l, kw), 0) for y in pre_years for l in LANGS)
        y2025_all_langs = sum(counts.get((VALIDATION_YEAR, l, kw), 0) for l in LANGS)
        y_g1_window_all_langs = sum(
            counts.get((y, l, kw), 0) for y in G1_VALIDATION_YEARS for l in LANGS
        )
        zero_16_20 = total_pre_all_langs == 0
        is_g1 = zero_16_20 and y_g1_window_all_langs >= MIN_VALIDATION_HITS_G1

        per_lang = {}
        for lang in LANGS:
            y_pre = [float(counts.get((y, lang, kw), 0)) for y in pre_years]
            y2025 = counts.get((VALIDATION_YEAR, lang, kw), 0)
            reg = LinearRegression(x_pre, y_pre)
            yhat = reg.predict(x_star)
            mean = sum(y_pre) / len(y_pre)
            B = max(yhat, mean, 0.0)
            per_lang[lang] = {"B": B, "y2025": y2025, "yhat": yhat, "mean": mean}

        result[kw] = {
            "zero_16_20": zero_16_20,
            "y2025_all_langs": y2025_all_langs,
            "y_g1_window_all_langs": y_g1_window_all_langs,
            "is_g1": is_g1,
            "per_lang": per_lang,
        }
    return result


def g2_keywords_at_c(kw_stats: dict, c: float) -> set[str]:
    """G2 = y2025 >= c*B AND y2025 >= 10, true in >=1 language -> propagates
    to the whole keyword. G1 keywords are excluded (judgment call #1, see
    module docstring)."""
    g2 = set()
    for kw, s in kw_stats.items():
        if s["is_g1"]:
            continue
        for stats in s["per_lang"].values():
            if (
                stats["y2025"] >= c * stats["B"]
                and stats["y2025"] >= MIN_VALIDATION_HITS_G2
                and stats["y2025"] >= 1
            ):
                g2.add(kw)
                break
    return g2


def run_grid_sweep(kw_stats: dict, per_ad_matches: list, total_2025_ads: int) -> pd.DataFrame:
    """Secondary calibration check (no manual labeling needed): for every
    (c, k) combination, how many keywords land in G2 and what share of ALL
    2025 ads (not just keyword-matching ones) get flagged. Rule C is a
    structural no-op with the current keyword list (see module docstring) -
    included as False for every ad, not omitted, so it's visible in the
    code that the term exists but can't fire."""
    g1_kws = {kw for kw, s in kw_stats.items() if s["is_g1"]}
    ads_2025 = [(ad_id, kws) for ad_id, year, kws in per_ad_matches if year == VALIDATION_YEAR]

    rows = []
    for c in C_GRID:
        g2_kws = g2_keywords_at_c(kw_stats, c)
        for k in K_GRID:
            flagged = 0
            for _ad_id, kws in ads_2025:
                n_g1 = sum(1 for kw in kws if kw in g1_kws)
                n_g2 = sum(1 for kw in kws if kw in g2_kws)
                rule_c = False  # structurally unreachable - see module docstring
                if n_g1 >= 1 or n_g2 >= k or rule_c:
                    flagged += 1
            rows.append({
                "c": c, "k": k,
                "n_g2_keywords": len(g2_kws),
                "n_g1_keywords": len(g1_kws),
                "flagged_2025": flagged,
                "total_2025_ads": total_2025_ads,
                "flagged_share_pct": 100 * flagged / total_2025_ads if total_2025_ads else 0.0,
            })
    return pd.DataFrame(rows)


def save_keyword_stats(kw_stats: dict, out_path: Path) -> None:
    rows = []
    for kw, s in kw_stats.items():
        row = {
            "keyword": kw,
            "zero_16_20": s["zero_16_20"],
            "y2025_all_langs": s["y2025_all_langs"],
            "is_g1": s["is_g1"],
        }
        for lang, stats in s["per_lang"].items():
            row[f"B_{lang}"] = round(stats["B"], 2)
            row[f"y2025_{lang}"] = stats["y2025"]
            row[f"multiple_{lang}"] = (
                round(stats["y2025"] / stats["B"], 2) if stats["B"] > 0 else None
            )
        rows.append(row)
    pd.DataFrame(rows).sort_values("keyword").to_csv(out_path, index=False)


def build_elbow_plot(sweep_df: pd.DataFrame, title_suffix: str):
    fig = px.line(
        sweep_df, x="c", y="flagged_share_pct", color="k",
        markers=True,
        title=f"v7 grid sweep: % of 2025 ads flagged vs. c, one line per k{title_suffix}",
        labels={"c": "c (multiple_g2_threshold)", "flagged_share_pct": "% of 2025 ads flagged", "k": "k"},
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="v7 scoring methodology - grid sweep")
    parser.add_argument("--config", type=str, default="pipeline/config.yaml")
    parser.add_argument("--exclude-layer3", action="store_true",
                         help="Exclude source=='LLM' (Layer 3 / layer3.py-derived) keywords")
    parser.add_argument("--include-bare-llm", action="store_true",
                         help="EXPERIMENTAL (Jeremias, 2026-08-03): also match bare "
                              "'LLM'/'LLMs' for the 'LLM' keyword, on top of the safe "
                              "spelled-out 'large language models' form. Does not touch "
                              "master_keywords.json or the live pipeline - pattern added "
                              "here only.")
    args = parser.parse_args()

    config = load_config(args.config)
    master_keywords = load_master_keywords()

    excluded_kws = set()
    label = "full_pool"
    title_suffix = " (full keyword pool)"
    if args.exclude_layer3:
        excluded_kws = {k["keyword"] for k in master_keywords if k["source"] == "LLM"}
        label = "no_layer3"
        title_suffix = f" (Layer 3 excluded, {len(excluded_kws)} keywords removed)"
        print(f"Excluding {len(excluded_kws)} Layer-3-sourced keywords: {sorted(excluded_kws)}")

    all_kws = [k["keyword"] for k in master_keywords if k["keyword"] not in excluded_kws]
    print(f"Running v7 scoring on {len(all_kws)} keywords ({label})...")

    en_patterns, local_patterns = build_keyword_patterns(master_keywords)

    extra_candidate_regex = None
    if args.include_bare_llm:
        llm_group = next(k["group"] for k in master_keywords if k["keyword"] == "LLM")
        en_patterns.append(("LLM", llm_group, re.compile(r"\bllms?\b")))
        extra_candidate_regex = r"\bllms?\b"
        label = "bare_llm" if label == "full_pool" else f"{label}_bare_llm"
        title_suffix += " + bare LLM experiment"
        print("  EXPERIMENTAL: bare 'LLM'/'LLMs' pattern added for the 'LLM' keyword.")

    counts, per_ad_matches = collect_matches(
        config, en_patterns, local_patterns, excluded_kws, extra_candidate_regex
    )

    print("  Classifying keywords (G1 fixed status + per-language benchmark stats)...")
    kw_stats = classify_keywords(counts, all_kws)
    n_g1 = sum(1 for s in kw_stats.values() if s["is_g1"])
    print(f"  {n_g1}/{len(all_kws)} keywords are G1 (zero 2016-2020 in all languages, "
          f">=1 hit in 2025 any language).")

    total_2025_ads = (
        pl.scan_parquet(Path(config.output_dir) / "classified_data" / "year=*" / "*.parquet")
        .filter(pl.col("tst_created").dt.year() == VALIDATION_YEAR)
        .select(pl.len())
        .collect(streaming=True)
        .item()
    )
    print(f"  Total 2025 ads (denominator): {total_2025_ads:,}")

    print("  Running c/k grid sweep...")
    sweep_df = run_grid_sweep(kw_stats, per_ad_matches, total_2025_ads)

    out_dir = Path(config.output_dir) / "v7_scoring"
    out_dir.mkdir(parents=True, exist_ok=True)

    kw_out = out_dir / f"keyword_stats_{label}.csv"
    save_keyword_stats(kw_stats, kw_out)
    print(f"  Saved keyword stats to {kw_out}")

    sweep_out = out_dir / f"grid_sweep_{label}.csv"
    sweep_df.to_csv(sweep_out, index=False)
    print(f"  Saved grid sweep table to {sweep_out}")

    fig = build_elbow_plot(sweep_df, title_suffix)
    plot_out = out_dir / f"grid_sweep_{label}.html"
    fig.write_html(plot_out)
    print(f"  Saved grid sweep plot to {plot_out}")

    print("\n=== Grid sweep summary ===")
    print(sweep_df.pivot(index="k", columns="c", values="flagged_share_pct").to_string())
    print("\n=== G2 keyword count per c ===")
    print(sweep_df.drop_duplicates("c")[["c", "n_g2_keywords"]].to_string(index=False))


if __name__ == "__main__":
    main()
