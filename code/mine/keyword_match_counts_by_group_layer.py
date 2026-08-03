"""
Keyword-level positive-match counts, per Jeremias (2026-08-02): how many of the
49 validated keywords (keyword_lists/master_keywords.json) actually produced at
least one match — not how many ADS matched, which is what every other sanity
script (occupation_genai_share_plot.py, group_strategy_analysis.py) reports.

Broken down two ways, both from the same table:
  - by Group (1/2/3, the validated priority tier)
  - by "layer" — Jeremias's own term for what master_keywords.json's `source`
    field already records: which of the three original keyword-list scripts a
    keyword came from (St -> keyword_lists/legacy/layer1_stanford.py, H&L ->
    legacy/layer2_hosseini.py, LLM -> legacy/layer3.py). Those layer*.py files
    are no longer read at match time (group_classification.py reads master_keywords.json,
    itself built from data/raw/validated_keywords.json — see
    build_master_keywords.py's docstring), but `source` is exactly the
    provenance Jeremias is asking about, so no new tracking was needed.

Match source: output/classified_data's `matched_group_keywords` column (List[
String] of triggering keyword names per ad, from group_classification.py's
_classify_file_worker) — NOT output/aggregation/keyword_time_series.parquet,
which is built from the older, currently-inactive output/scored_data /
keyword_scoring.py path and would give stale numbers.

Counting convention: per keyword, the number of ADS it matched at least one
occurrence in (i.e. counting each ad once per keyword, not raw regex hits) -
matches the ad-level semantics `matched_group_keywords` already encodes.
"""

import argparse
from pathlib import Path

import polars as pl

from pipeline.config_loader import load_config
from pipeline.group_classification import load_master_keywords

LAYER_LABELS = {
    "St": "St (layer1_stanford.py)",
    "H&L": "H&L (layer2_hosseini.py)",
    "LLM": "LLM (layer3.py)",
}


def main():
    parser = argparse.ArgumentParser(description="Per-keyword positive-match counts by group/layer")
    parser.add_argument("--config", type=str, default="pipeline/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    master_keywords = load_master_keywords()
    master_df = pl.DataFrame([
        {"keyword": k["keyword"], "group": k["group"], "source": k["source"]}
        for k in master_keywords
    ])
    print(f"Loaded {len(master_df)} validated keywords from master_keywords.json.")

    lf = pl.scan_parquet(Path(config.output_dir) / "classified_data" / "year=*" / "*.parquet")
    counts = (
        lf.select("matched_group_keywords")
        .explode("matched_group_keywords")
        .drop_nulls()
        .rename({"matched_group_keywords": "keyword"})
        .group_by("keyword")
        .agg(pl.len().alias("ad_match_count"))
        .collect()
    )

    table = (
        master_df.join(counts, on="keyword", how="left")
        .with_columns(pl.col("ad_match_count").fill_null(0))
        .sort(["group", "ad_match_count"], descending=[False, True])
    )

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "keyword_match_counts_by_group_layer.csv"
    table.write_csv(out_path)
    print(f"\nFull per-keyword table saved to {out_path}\n")
    print(table.to_pandas().to_string(index=False))

    matched = table.filter(pl.col("ad_match_count") > 0)

    print("\n=== Keywords with >=1 match, by Group ===")
    for grp in sorted(table["group"].unique().to_list()):
        total_in_group = table.filter(pl.col("group") == grp).height
        matched_in_group = matched.filter(pl.col("group") == grp).height
        print(f"  Group {grp}: {matched_in_group}/{total_in_group} keywords matched at least once")

    print("\n=== Keywords with >=1 match, by layer (source) ===")
    for src in ["St", "H&L", "LLM"]:
        total_in_src = table.filter(pl.col("source") == src).height
        matched_in_src = matched.filter(pl.col("source") == src).height
        print(f"  {LAYER_LABELS[src]}: {matched_in_src}/{total_in_src} keywords matched at least once")

    print("\n=== Cross-tab: keywords with >=1 match, Group x layer ===")
    header = f"  {'Group':<8}" + "".join(f"{LAYER_LABELS[s]:>26}" for s in ["St", "H&L", "LLM"]) + f"{'Total':>10}"
    print(header)
    for grp in sorted(table["group"].unique().to_list()):
        row = f"  {grp:<8}"
        row_total_matched = 0
        row_total_all = 0
        for src in ["St", "H&L", "LLM"]:
            cell_all = table.filter((pl.col("group") == grp) & (pl.col("source") == src)).height
            cell_matched = matched.filter((pl.col("group") == grp) & (pl.col("source") == src)).height
            row += f"{f'{cell_matched}/{cell_all}':>26}"
            row_total_matched += cell_matched
            row_total_all += cell_all
        row += f"{f'{row_total_matched}/{row_total_all}':>10}"
        print(row)

    overall_matched, overall_total = matched.height, table.height
    print(f"\nOverall: {overall_matched}/{overall_total} validated keywords produced >=1 match "
          f"across the full classified dataset.")


if __name__ == "__main__":
    main()
