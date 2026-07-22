"""
Occupation sanity-check plot for pipeline/group_classification.py's output
(similar to thesis Figure 2/6: top occupations by GenAI posting share).

Two-hop join: ad occupations[].id (x28 occupation id) -> AVAM code
(data/raw/x28_Occupation_to_AVAM.xlsx) -> ISCO code (data/raw/q93_isco.csv).
Ads are grouped by ISCO code (the thesis's own occupational classification,
Section 3.4) rather than raw x28 occupation id, since many x28 ids are just
spelling/title variants of the same underlying occupation and would otherwise
fragment the plot into near-duplicates. Each ISCO group is labeled with the
first x28 occupation name found for it in the mapping file - a simple,
deterministic label choice, not the official ISCO-08 title text (that isn't
in any of our source files, only the numeric code is).

This is a SANITY CHECK, not a publication figure: results should look broadly
similar in shape to the thesis's own findings (concentrated in IT/research/
professional occupations). Anything that looks obviously wrong (e.g. a small,
low-relevance occupation dominating) is flagged in the printed output, not
just silently plotted.
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import polars as pl

from pipeline.config_loader import load_config

MIN_ADS_PER_OCCUPATION = 20  # drop occupations with too few ads to be meaningful
TOP_N = 20


def build_occupation_to_isco_map(root: Path) -> dict:
    """Returns {x28_occupation_id: (isco_code, display_label)}."""
    avam_map = pd.read_excel(root / "data" / "raw" / "x28_Occupation_to_AVAM.xlsx")
    isco_map = pd.read_csv(root / "data" / "raw" / "q93_isco.csv")

    avam_map = avam_map.rename(columns={
        "id (x28)": "x28_id", "name (x28)": "x28_name", "code (AVAM)": "avam_code",
    })
    # Prefer the newer AVAM-2020 catalogue row when an x28 id has both.
    avam_map = avam_map.sort_values("name (Catalogue)", ascending=False)
    avam_map = avam_map.drop_duplicates(subset="x28_id", keep="first")

    isco_map = isco_map.rename(columns={"avam": "avam_code", "isco": "isco_code"})

    merged = avam_map.merge(isco_map, on="avam_code", how="inner")

    mapping = {}
    for row in merged.itertuples():
        if row.x28_id not in mapping:
            mapping[row.x28_id] = (str(row.isco_code), row.x28_name)
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Occupation GenAI-share sanity plot")
    parser.add_argument("--config", type=str, default="pipeline/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(__file__).resolve().parents[2]

    occ_map = build_occupation_to_isco_map(root)
    print(f"Loaded occupation -> ISCO mapping for {len(occ_map)} x28 occupation ids.")

    lf = pl.scan_parquet(Path(config.output_dir) / "classified_data" / "year=*" / "*.parquet")
    df = lf.select(["group", "occupations"]).collect()

    rows = []
    for group, occs in zip(df["group"], df["occupations"]):
        has_group = group != "NA"
        if occs is None:
            continue
        for occ in occs:
            occ_id = occ["id"]
            if occ_id in occ_map:
                isco_code, label = occ_map[occ_id]
                rows.append((isco_code, label, has_group))

    agg = pd.DataFrame(rows, columns=["isco_code", "label", "has_group"])
    if agg.empty:
        print("No ads had an occupation matching the AVAM/ISCO mapping - nothing to plot.")
        return

    summary = (
        agg.groupby(["isco_code", "label"])
        .agg(total_ads=("has_group", "size"), flagged_ads=("has_group", "sum"))
        .reset_index()
    )
    summary = summary[summary["total_ads"] >= MIN_ADS_PER_OCCUPATION]
    summary["share"] = summary["flagged_ads"] / summary["total_ads"]
    summary = summary.sort_values("share", ascending=False)

    print(f"\n{len(summary)} occupations with >= {MIN_ADS_PER_OCCUPATION} ads.")
    print("Top 5 by GenAI share:")
    print(summary.head(5)[["label", "isco_code", "total_ads", "share"]].to_string(index=False))

    if not summary.empty:
        top_row = summary.iloc[0]
        if top_row["share"] > 0.5 and top_row["total_ads"] < 100:
            print(f"\nFLAG: top occupation '{top_row['label']}' has a high share "
                  f"({top_row['share']:.0%}) on a small sample ({top_row['total_ads']} ads) - "
                  f"could be noise, worth checking manually before trusting the plot.")

    top = summary.head(TOP_N).sort_values("share")
    top["occupation"] = top["label"] + " (" + top["isco_code"] + ")"
    fig = px.bar(
        top.sort_values("share"), x="share", y="occupation", orientation="h",
        labels={"share": "Share of ads with any Group 1/2/3 keyword match", "occupation": ""},
        title=f"Top {TOP_N} occupations by GenAI posting share (sanity check)",
    )
    fig.update_layout(height=700, width=1000)

    out_dir = Path(config.output_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "occupation_genai_share.png"
    fig.write_image(out_path)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
