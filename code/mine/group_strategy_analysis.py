"""
Small-scale (but complete-population) test of the three Group 1/2/3 combination
strategies proposed in notes/group_strategy_proposals_for_claude_code.md.

WHY THIS DOESN'T NEED A FULL RESCAN: output/classified_data's `group` column already
tells us which ads have ZERO keyword matches across all three groups - those are
exactly the ones with group == "NA" (see group_classification.py's
_classify_file_worker: an ad only gets "NA" when classify_text() returns an empty
dict). So the ads where group != "NA" are the COMPLETE population of ads with any
match at all, not a sample - just under 3.6M rows filtered down to ~6.6K. Only those
need reclassifying to recover per-group keyword counts (the persisted output only kept
the single best-priority group's keywords, per group_classification.py's own
docstring/comments - Options B and C both need all three groups' counts, not just the
winner).

Re-runs classify_text() (imported from pipeline.group_classification, not
reimplemented) on just those ~6.6K ads to get the FULL {keyword: group} match dict per
ad, then evaluates:
  - Option A: two confidence tiers (>=1 G1 vs >=1 G2/G3)
  - Option B: >=1 G1 OR >=k combined G2+G3 distinct keywords, swept k=2/3/4
  - Option C: weighted score 3*G1+2*G2+G3, swept cutoff, weights fixed
per notes/group_strategy_proposals_for_claude_code.md. Also reports, for ads newly
flagged by B/C that aren't already Group 1, which keywords are driving it - the
LLM/RAG-false-positive check the proposal doc explicitly asks for.

Counting convention: DISTINCT keyword count per group (a set, not raw occurrences) -
matches the existing (dormant) Rule 3 precedent in pipeline/keyword_scoring.py.
"""

import argparse
from collections import Counter
from pathlib import Path

import plotly.express as px
import polars as pl

from pipeline.config_loader import load_config
from pipeline.group_classification import (
    build_keyword_patterns,
    classify_text,
    load_master_keywords,
    normalize,
)

B_THRESHOLDS = [2, 3, 4, 5, 6]
C_WEIGHTS = {1: 3, 2: 2, 3: 1}
C_CUTOFFS = [3, 4, 5, 6]


def recompute_full_matches(df: pl.DataFrame, en_patterns, local_patterns) -> pl.DataFrame:
    rows = []
    for ad_id, text, lang, persisted_group in zip(
        df["ad_id"], df["content_clean"], df["detected_language"], df["group"]
    ):
        norm = normalize(text or "")
        matched = classify_text(norm, lang, en_patterns, local_patterns)
        g1 = {k for k, g in matched.items() if g == 1}
        g2 = {k for k, g in matched.items() if g == 2}
        g3 = {k for k, g in matched.items() if g == 3}

        # Sanity check (per plan's verification step): the persisted best-group
        # keyword(s) must appear inside the corresponding recomputed set.
        best = {"1": g1, "2": g2, "3": g3}.get(persisted_group)
        if best is not None and not best:
            raise AssertionError(
                f"ad_id={ad_id}: persisted group={persisted_group} but recomputed "
                f"group-{persisted_group} keyword set is empty - reclassification "
                f"disagrees with the trusted persisted output."
            )

        rows.append({
            "ad_id": ad_id,
            "g1_count": len(g1), "g2_count": len(g2), "g3_count": len(g3),
            "g1_kw": sorted(g1), "g2_kw": sorted(g2), "g3_kw": sorted(g3),
        })
    return pl.DataFrame(rows)


def keyword_freq(df: pl.DataFrame, mask: pl.Series, cols=("g2_kw", "g3_kw")) -> Counter:
    c = Counter()
    sub = df.filter(mask)
    for col in cols:
        for kw_list in sub[col]:
            c.update(kw_list)
    return c


def main():
    parser = argparse.ArgumentParser(description="Test Group 1/2/3 combination strategies")
    parser.add_argument("--config", type=str, default="pipeline/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    classified_root = Path(config.output_dir) / "classified_data"

    print("Loading ads with any keyword match (group != NA)...")
    lf = pl.scan_parquet(classified_root / "year=*" / "*.parquet")
    df = (
        lf.filter(pl.col("group") != "NA")
        .select(["ad_id", "content_clean", "detected_language", "group"])
        .collect()
    )
    print(f"  {df.height:,} ads loaded "
          f"(expected 2348+777+3471=6596 from the last confirmed post-fix run).")

    en_patterns, local_patterns = build_keyword_patterns(load_master_keywords())
    print("Recomputing full per-group keyword matches...")
    matches = recompute_full_matches(df, en_patterns, local_patterns)
    print("  Sanity check passed: every persisted best-group keyword set is non-empty "
          "in the recomputed data.\n")

    g1, g2, g3 = matches["g1_count"], matches["g2_count"], matches["g3_count"]
    g23 = g2 + g3
    total_population = matches.height

    print("=" * 70)
    print(f"POPULATION: {total_population:,} ads with >=1 keyword match "
          f"(out of ~3.64M total ads scanned)")
    print("=" * 70)

    # --- Option A: two confidence tiers ---
    high_conf = (g1 >= 1)
    possible_raw = (g23 >= 1)
    possible_only = possible_raw & ~high_conf
    print("\n--- Option A: Two Confidence Tiers ---")
    print(f"  High-confidence (>=1 Group 1):            {high_conf.sum():,}")
    print(f"  Possible (>=1 Group 2/3, raw definition):  {possible_raw.sum():,}")
    print(f"  Possible-only (excludes high-confidence):  {possible_only.sum():,}")
    print(f"  Overlap (high-conf ads that ALSO have G2/3): {(high_conf & possible_raw).sum():,}")

    # --- Option B: corroboration count sweep ---
    print("\n--- Option B: Corroboration Count (>=1 G1 OR >=k combined G2+G3) ---")
    b_flags = {}
    for k in B_THRESHOLDS:
        flag = (g1 >= 1) | (g23 >= k)
        b_flags[k] = flag
        newly = flag & ~high_conf
        print(f"  k={k}: {flag.sum():,} flagged total "
              f"({newly.sum():,} newly swept in beyond Group 1 alone)")

    elbow_df = pl.DataFrame({
        "k": B_THRESHOLDS,
        "total_flagged": [int(b_flags[k].sum()) for k in B_THRESHOLDS],
    })
    elbow_fig = px.line(
        elbow_df, x="k", y="total_flagged", markers=True,
        title="Option B elbow plot: total ads flagged vs. corroboration threshold k",
        labels={"k": "k (min combined Group 2+3 keywords, when no Group 1 match)",
                "total_flagged": "Total ads flagged"},
    )
    elbow_out = Path(config.output_dir) / "plots" / "option_b_elbow.html"
    elbow_out.parent.mkdir(parents=True, exist_ok=True)
    elbow_fig.write_html(elbow_out)
    print(f"  Saved elbow plot to {elbow_out.resolve()}")

    # --- Option C: weighted score sweep ---
    print(f"\n--- Option C: Weighted Score (weights fixed: "
          f"G1={C_WEIGHTS[1]}, G2={C_WEIGHTS[2]}, G3={C_WEIGHTS[3]}) ---")
    score = C_WEIGHTS[1] * g1 + C_WEIGHTS[2] * g2 + C_WEIGHTS[3] * g3
    c_flags = {}
    for cutoff in C_CUTOFFS:
        flag = score >= cutoff
        c_flags[cutoff] = flag
        newly = flag & ~high_conf
        print(f"  cutoff={cutoff}: {flag.sum():,} flagged total "
              f"({newly.sum():,} newly swept in beyond Group 1 alone)")

    # --- Overlap between headline variants ---
    b_headline = b_flags[2]
    c_headline = c_flags[4]
    print("\n--- Overlap between headline variants (A=high-conf, B@k=2, C@cutoff=4) ---")
    print(f"  High-conf only (A):        {(high_conf & ~b_headline & ~c_headline).sum():,}")
    print(f"  B@2 only:                  {(~high_conf & b_headline & ~c_headline).sum():,}")
    print(f"  C@4 only:                  {(~high_conf & ~b_headline & c_headline).sum():,}")
    print(f"  All three agree flagged:   {(high_conf | b_headline | c_headline).sum():,} (union)")
    print(f"  All three agree (intersect of A, B@2, C@4): "
          f"{(high_conf & b_headline & c_headline).sum():,}")
    print(f"  B@2 and C@4 agree exactly: {(b_headline == c_headline).sum():,} / {total_population:,} ads")

    # --- Newly-swept-in keyword frequency (false-positive check) ---
    print("\n--- Keywords driving the 'newly swept in' ads (B@2, not already Group 1) ---")
    newly_b2 = b_headline & ~high_conf
    freq_b2 = keyword_freq(matches, newly_b2)
    for kw, count in freq_b2.most_common(15):
        print(f"  {kw:30s} {count:>5,} ads")
    if freq_b2:
        top_kw, top_count = freq_b2.most_common(1)[0]
        share = top_count / newly_b2.sum() if newly_b2.sum() else 0
        if share > 0.5:
            print(f"  FLAG: '{top_kw}' alone drives {share:.0%} of B@2's newly-flagged "
                  f"ads - worth a manual eyeball before trusting this option.")

    print("\n--- Keywords driving the 'newly swept in' ads (C@4, not already Group 1) ---")
    newly_c4 = c_headline & ~high_conf
    freq_c4 = keyword_freq(matches, newly_c4)
    for kw, count in freq_c4.most_common(15):
        print(f"  {kw:30s} {count:>5,} ads")
    if freq_c4:
        top_kw, top_count = freq_c4.most_common(1)[0]
        share = top_count / newly_c4.sum() if newly_c4.sum() else 0
        if share > 0.5:
            print(f"  FLAG: '{top_kw}' alone drives {share:.0%} of C@4's newly-flagged "
                  f"ads - worth a manual eyeball before trusting this option.")

    # --- Save a copy to output/ for reference ---
    out_dir = Path(config.output_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "group_strategy_analysis.csv"
    matches.with_columns([
        pl.col("g1_kw").list.join(";"),
        pl.col("g2_kw").list.join(";"),
        pl.col("g3_kw").list.join(";"),
        high_conf.alias("option_a_high_confidence"),
        possible_raw.alias("option_a_possible_raw"),
        b_flags[2].alias("option_b_k2"),
        b_flags[3].alias("option_b_k3"),
        b_flags[4].alias("option_b_k4"),
        c_flags[4].alias("option_c_cutoff4"),
        score.alias("option_c_score"),
    ]).write_csv(out_path)
    print(f"\nSaved per-ad results to {out_path.resolve()}")


if __name__ == "__main__":
    main()
