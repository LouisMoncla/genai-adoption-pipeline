"""
Sanity-check printout for pipeline/group_classification.py's output.

Prints 3-5 example matched ads per group (text + which keyword(s) triggered the
match), the way Domenico describes doing a manual audit in Appendix D.4 - so the
result can be eyeballed for genuine GenAI content vs. false positives before
trusting the classification.

Run after classify_postings_by_group() has produced output/classified_data/.
"""

import argparse
from pathlib import Path

import polars as pl

from pipeline.config_loader import load_config

EXAMPLES_PER_GROUP = 5
TEXT_PREVIEW_CHARS = 300


def main():
    parser = argparse.ArgumentParser(description="Print example classified ads per group")
    parser.add_argument("--config", type=str, default="pipeline/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    classified_root = Path(config.output_dir) / "classified_data"

    lf = pl.scan_parquet(classified_root / "year=*" / "*.parquet")

    counts = (
        lf.group_by("group").agg(pl.len().alias("count")).collect().sort("group")
    )
    total = counts["count"].sum()
    print("=== Classification counts ===")
    for row in counts.iter_rows(named=True):
        pct = 100 * row["count"] / total if total else 0
        print(f"  group {row['group']:>3}: {row['count']:>10,} ads ({pct:5.1f}%)")
    print()

    for group in ("1", "2", "3"):
        print(f"=== Group {group} - {EXAMPLES_PER_GROUP} example ads ===")
        sample = (
            lf.filter(pl.col("group") == group)
            .select(["ad_id", config.col.content, "detected_language", "matched_group_keywords"])
            .limit(200)  # pull a small pool, then sample from it in-memory
            .collect()
        )
        if sample.is_empty():
            print("  (no ads in this group)\n")
            continue

        n = min(EXAMPLES_PER_GROUP, sample.height)
        examples = sample.sample(n=n, seed=42)

        for row in examples.iter_rows(named=True):
            text = (row[config.col.content] or "").strip().replace("\n", " ")
            preview = text[:TEXT_PREVIEW_CHARS] + ("..." if len(text) > TEXT_PREVIEW_CHARS else "")
            print(f"  ad_id={row['ad_id']}  lang={row['detected_language']}")
            print(f"  triggered by: {row['matched_group_keywords']}")
            print(f"  text: {preview}")
            print()
        print()


if __name__ == "__main__":
    main()
