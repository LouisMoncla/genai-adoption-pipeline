"""
Occupation sanity-check plot for pipeline/group_classification.py's output
(similar to thesis Figure 2/6: top occupations by GenAI posting share).

Two-hop join: ad occupations[].id (x28 occupation id) -> AVAM code
(data/raw/x28_Occupation_to_AVAM.xlsx) -> ISCO code (data/raw/q93_isco.csv).
Ads are grouped by ISCO code (the thesis's own occupational classification,
Section 3.4) rather than raw x28 occupation id, since many x28 ids are just
spelling/title variants of the same underlying occupation and would otherwise
fragment the plot into near-duplicates. Each ISCO group is labeled with the
official group name from data/raw/isco_structure.csv (added 2026-07-23,
superseding an even earlier version that used an arbitrary first-encountered
x28 occupation name, usually German - see git history).

isco_structure.csv is the official ISCO structure table (one canonical
"description" per unit-group code, plus the full major/sub_major/minor
hierarchy) - cleaner than an alternate-title index since there's no
"which of these N variants do we pick" ambiguity. IMPORTANT: the raw file
mixes FOUR classification eras under the same numeric codes (ISCO_version
column has ISCO-08/88/68/58) - the same 4-digit code means different things
across versions (e.g. unit 1120 is "Managing Directors and Chief Executives"
under ISCO-08 but "Senior government officials" under an older version).
Must filter to ISCO_version == "ISCO-08" only, which gives exactly 436
unique codes with zero inconsistent descriptions - confirmed by checking
before wiring this in, since silently mixing eras would have corrupted the
mapping without any obvious symptom.

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


def _load_isco_titles(root: Path) -> dict:
    """Returns {isco_code (int): official ISCO-08 unit-group name (str)}."""
    structure = pd.read_csv(root / "data" / "raw" / "isco_structure.csv", encoding="cp1252")
    structure = structure[structure["ISCO_version"] == "ISCO-08"]
    return dict(zip(structure["unit"].astype(int), structure["description"].str.strip()))


def build_occupation_to_isco_map(root: Path) -> dict:
    """Returns {x28_occupation_id: (isco_code, display_label)}."""
    avam_map = pd.read_excel(root / "data" / "raw" / "x28_Occupation_to_AVAM.xlsx")
    isco_map = pd.read_csv(root / "data" / "raw" / "q93_isco.csv")
    isco_titles = _load_isco_titles(root)

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
            # Fall back to the x28 name only if this ISCO code isn't in the
            # index (shouldn't happen often - coverage checked at 437 codes).
            label = isco_titles.get(row.isco_code, row.x28_name)
            mapping[row.x28_id] = (str(row.isco_code), label)
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
            # occupations[].id comes out of the export as a string ("11001162"),
            # but occ_map's keys are int (pandas' Excel read infers the AVAM
            # mapping's id column as numeric) - every lookup silently failed
            # until this cast, which is why the first run found zero matches.
            try:
                occ_id = int(occ["id"])
            except (TypeError, ValueError):
                continue
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
