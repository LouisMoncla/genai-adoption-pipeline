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

# Per Jeremias, 2026-07-30: the "Fotomodell" (photo model) x28 occupation id was
# found (2026-07-29 investigation) to disproportionately co-occur with genuine
# AI/ML job ads - 8.14% of Fotomodell-tagged ads are GenAI-flagged vs the ~0.2%
# corpus-wide base rate, and every sampled Group-3 example was a real ML/AI
# posting (Master's thesis in Geospatial AI, Roche MLOps internship, etc.), not
# an actual photo-modeling ad. Root cause: x28's own occupation-tagging seems to
# fuzzy-match ads whose text is saturated with "model"/"Foundation Model"/
# "Language Model" mentions to AVAM-2020 code 101924, literally named "Model",
# alongside the ad's real (correct) occupation tag. Jeremias's fix: only trust
# the Fotomodell tag when it's the ad's SOLE occupation entry - if an ad has
# another occupation tag too, that other one is almost certainly the real job
# title and Fotomodell is the mistagging artifact, so drop it.
FOTOMODELL_X28_ID = 11001610


def _drop_secondary_fotomodell_tag(occs: list) -> list:
    """If Fotomodell co-occurs with >=1 other occupation on the same ad, drop
    the Fotomodell entry and keep the rest - only trust it when it's the ad's
    only occupation tag."""
    if len(occs) <= 1:
        return occs
    try:
        has_other = any(int(o["id"]) != FOTOMODELL_X28_ID for o in occs)
    except (TypeError, ValueError):
        return occs
    if not has_other:
        return occs
    return [o for o in occs if _safe_id(o) != FOTOMODELL_X28_ID]


def _safe_id(occ) -> int | None:
    try:
        return int(occ["id"])
    except (TypeError, ValueError):
        return None

# Ads tagged with more than this many occupation ids have an ambiguous occupation -
# exclude them from the occupation-share breakdown (not from any other count). Matches
# Jeremias's own R scripts (code/from_jeremias/test_ads_isco3_timeseries.R and
# wfh_analysis.R), both of which apply the identical MAX_JOBS <- 5L cutoff before their
# own x28->AVAM->ISCO join, for the same reason: with >5 tags, which one the ad
# "actually" is becomes unclear enough that keeping it would add noise, not signal.
MAX_OCCUPATIONS_PER_AD = 5


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
    ambiguous_skipped = 0
    fotomodell_dropped = 0
    for group, occs in zip(df["group"], df["occupations"]):
        has_group = group != "NA"
        if occs is None:
            continue
        filtered = _drop_secondary_fotomodell_tag(occs)
        if len(filtered) != len(occs):
            fotomodell_dropped += 1
        occs = filtered
        if len(occs) > MAX_OCCUPATIONS_PER_AD:
            ambiguous_skipped += 1
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

    print(f"Dropped a secondary 'Fotomodell' tag on {fotomodell_dropped:,} ad(s) that also "
          f"had another occupation tag (per Jeremias's fix - see FOTOMODELL_X28_ID comment).")
    print(f"Excluded {ambiguous_skipped:,} ad(s) with >{MAX_OCCUPATIONS_PER_AD} occupation "
          f"tags (ambiguous occupation) from the occupation-share breakdown.")

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
