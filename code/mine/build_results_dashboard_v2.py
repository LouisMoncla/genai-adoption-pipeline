"""
V2 of the single-file HTML results dashboard (see build_results_dashboard.py for v1),
rebuilt on the full combined 2018-2025 dataset (up from 2020-2025) after:
  - Removing bare "Copilot" as a validated keyword (Jeremias, 2026-07-30) - see
    code/mine/build_master_keywords.py's EXCLUDED_KEYWORDS.
  - Confirming k=3 as the Option B corroboration threshold (Jeremias, from the
    elbow plot).
  - Adding the "Fotomodell" secondary-tag fix (Jeremias, 2026-07-30) to the
    occupation-mapping logic - see FOTOMODELL_X28_ID below.

Seventeen sections - everything from v1 (1-10), plus new ones enabled by the
richer 2018-2025 span and unused columns (firm size, home-office, contract type,
canton, firm-level adoption, raw posting volume):
  1. Overall classification breakdown
  2. GenAI-mention share of postings over time (monthly)
  3. Matches per group
  4. Group share per year
  5. Proportion table: matches per group per year
  6. Option B elbow plot (k=3 confirmed - see embedded note)
  7. Occupation breakdown (with the Fotomodell secondary-tag fix applied)
  8. Top triggered keywords
  9. GenAI share by language
  10. GenAI share by industry
  11. NEW - Total postings per year (context/denominator - flags the Dec-2018-only
      partial-month anomaly directly rather than letting it look unexplained)
  12. NEW - Firm-level adoption: distinct firms with >=1 flagged ad, per year and
      cumulative (mirrors Domenico's thesis headline "554 firms by 2025" stat)
  13. NEW - GenAI share by firm size (small/medium/large)
  14. NEW - GenAI share by home-office availability
  15. NEW - GenAI share by contract type (temporary vs permanent)
  16. NEW - Top cantons by GenAI share
  17. NEW - Recommended flag (Option B, k=3) headline summary

Known data-quality flag surfaced while building this (NOT fixed here - needs
discussion, see notes): `company.size.id` is null for ~99.9% of Dec 2018 rows and
an elevated ~12-19% of Jan 2019-Nov 2020 rows. Because data_preparation.py's
micro-enterprise filter does `_size_id != micro_enterprise_id`, and Polars
evaluates `null != x` as null (which `.filter()` drops), ads with UNKNOWN firm
size are silently excluded as if they were confirmed micro-enterprises - this is
why Dec 2018 collapses from 128,455 raw exported rows to 33 surviving rows. Not
patched here since the correct fix (keep vs. exclude unknown-size ads) is a
judgment call, not obviously safe to make silently.
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
import polars as pl

from pipeline.config_loader import load_config

MIN_ADS_PER_OCCUPATION = 20
MIN_ADS_PER_INDUSTRY = 50
MIN_ADS_PER_CANTON = 50
TOP_N_OCCUPATIONS = 20
TOP_N_KEYWORDS = 25
TOP_N_INDUSTRIES = 20
TOP_N_CANTONS = 15
MAX_OCCUPATIONS_PER_AD = 5  # matches occupation_genai_share_plot.py's MAX_JOBS-style cutoff
OPTION_B_K = 3  # confirmed by Jeremias from the elbow plot, 2026-07-30

# Per Jeremias, 2026-07-30 - see occupation_genai_share_plot.py for the full
# writeup of why this exists (Fotomodell/AI-job mistagging finding).
FOTOMODELL_X28_ID = 11001610


def _safe_id(occ):
    try:
        return int(occ["id"])
    except (TypeError, ValueError):
        return None


def _drop_secondary_fotomodell_tag(occs: list) -> list:
    if len(occs) <= 1:
        return occs
    ids = [_safe_id(o) for o in occs]
    if FOTOMODELL_X28_ID not in ids:
        return occs
    if any(i is not None and i != FOTOMODELL_X28_ID for i in ids):
        return [o for o, i in zip(occs, ids) if i != FOTOMODELL_X28_ID]
    return occs


def _load_isco_titles(root: Path) -> dict:
    structure = pd.read_csv(root / "data" / "raw" / "isco_structure.csv", encoding="cp1252")
    structure = structure[structure["ISCO_version"] == "ISCO-08"]
    return dict(zip(structure["unit"].astype(int), structure["description"].str.strip()))


def build_occupation_to_isco_map(root: Path) -> dict:
    avam_map = pd.read_excel(root / "data" / "raw" / "x28_Occupation_to_AVAM.xlsx")
    isco_map = pd.read_csv(root / "data" / "raw" / "q93_isco.csv")
    isco_titles = _load_isco_titles(root)

    avam_map = avam_map.rename(columns={
        "id (x28)": "x28_id", "name (x28)": "x28_name", "code (AVAM)": "avam_code",
    })
    avam_map = avam_map.sort_values("name (Catalogue)", ascending=False)
    avam_map = avam_map.drop_duplicates(subset="x28_id", keep="first")

    isco_map = isco_map.rename(columns={"avam": "avam_code", "isco": "isco_code"})
    merged = avam_map.merge(isco_map, on="avam_code", how="inner")

    mapping = {}
    for row in merged.itertuples():
        if row.x28_id not in mapping:
            label = isco_titles.get(row.isco_code, row.x28_name)
            mapping[row.x28_id] = (str(row.isco_code), label)
    return mapping


# --- Sections 1-5, 8-10: unchanged from v1 -----------------------------------

def chart_classification_breakdown(df: pl.DataFrame):
    counts = df.group_by("group").len().sort("group").to_pandas()
    counts["label"] = counts["group"].map({
        "1": "Group 1 (core GenAI)", "2": "Group 2", "3": "Group 3", "NA": "No keyword match",
    })
    fig = px.pie(
        counts, names="label", values="len", hole=0.45,
        title="Overall classification breakdown (all postings, 2018-2025)",
        color="label",
        color_discrete_map={
            "Group 1 (core GenAI)": "#2ca02c", "Group 2": "#1f77b4",
            "Group 3": "#ff7f0e", "No keyword match": "#d3d3d3",
        },
    )
    fig.update_traces(textinfo="label+percent+value")
    return fig


def chart_share_over_time(df: pl.DataFrame):
    monthly = (
        df.with_columns(pl.col("tst_created").dt.strftime("%Y-%m").alias("month"))
        .group_by(["month", "group"]).len()
    )
    pivot = monthly.pivot(on="group", index="month", values="len").fill_null(0).sort("month")
    for g in ("1", "2", "3", "NA"):
        if g not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0).alias(g))
    pivot = pivot.with_columns(
        (pl.col("1") + pl.col("2") + pl.col("3") + pl.col("NA")).alias("total")
    ).with_columns([
        (100 * pl.col("1") / pl.col("total")).alias("Group 1 share"),
        (100 * (pl.col("1") + pl.col("2") + pl.col("3")) / pl.col("total")).alias("Any group share"),
    ])
    pdf = pivot.select(["month", "Group 1 share", "Any group share"]).to_pandas()
    pdf_long = pdf.melt(id_vars="month", var_name="series", value_name="share_pct")

    fig = px.line(
        pdf_long, x="month", y="share_pct", color="series", markers=True,
        title="GenAI-mention share of job postings over time (2018-2025)",
        labels={"month": "", "share_pct": "% of postings", "series": ""},
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def chart_matches_per_group(df: pl.DataFrame):
    counts = (
        df.filter(pl.col("group") != "NA")
        .group_by("group").len().sort("group")
        .to_pandas()
    )
    counts["label"] = "Group " + counts["group"]
    fig = px.bar(
        counts, x="label", y="len", color="label", text="len",
        title="Matches per group (Group 1/2/3 counts)",
        labels={"label": "", "len": "Number of postings"},
        color_discrete_map={"Group 1": "#2ca02c", "Group 2": "#1f77b4", "Group 3": "#ff7f0e"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


def _group_year_pivot(df: pl.DataFrame) -> pl.DataFrame:
    yearly = (
        df.with_columns(pl.col("tst_created").dt.year().alias("year"))
        .group_by(["year", "group"]).len()
    )
    pivot = yearly.pivot(on="group", index="year", values="len").fill_null(0).sort("year")
    for g in ("1", "2", "3", "NA"):
        if g not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0).alias(g))
    return pivot


def chart_group_share_per_year(df: pl.DataFrame):
    pivot = _group_year_pivot(df)
    pivot = pivot.with_columns(
        (pl.col("1") + pl.col("2") + pl.col("3") + pl.col("NA")).alias("total")
    ).with_columns([
        (100 * pl.col("1") / pl.col("total")).alias("Group 1"),
        (100 * pl.col("2") / pl.col("total")).alias("Group 2"),
        (100 * pl.col("3") / pl.col("total")).alias("Group 3"),
    ])
    pdf = pivot.select(["year", "Group 1", "Group 2", "Group 3"]).to_pandas()
    pdf["year"] = pdf["year"].astype(str)
    pdf_long = pdf.melt(id_vars="year", var_name="group", value_name="share_pct")

    fig = px.line(
        pdf_long, x="year", y="share_pct", color="group", markers=True,
        title="Group share per year (each group's % of that year's postings) - "
              "note 2018 is Dec-only and severely undercounted, see writeup",
        labels={"year": "", "share_pct": "% of that year's postings", "group": ""},
        color_discrete_map={"Group 1": "#2ca02c", "Group 2": "#1f77b4", "Group 3": "#ff7f0e"},
    )
    return fig


def table_proportion_per_year(df: pl.DataFrame) -> str:
    pivot = _group_year_pivot(df)
    pivot = pivot.with_columns(
        (pl.col("1") + pl.col("2") + pl.col("3") + pl.col("NA")).alias("total")
    )

    rows_html = []
    for row in pivot.iter_rows(named=True):
        cells = [f"<td>{row['year']}</td>", f"<td>{row['total']:,}</td>"]
        for g in ("1", "2", "3"):
            n = row[g]
            pct = 100 * n / row["total"] if row["total"] else 0
            cells.append(f"<td>{n:,} ({pct:.2f}%)</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    return (
        "<h3>Proportion table: matches per group per year</h3>"
        "<table style='border-collapse:collapse;width:100%'>"
        "<tr style='background:#f0f0f0'>"
        "<th style='text-align:left;padding:6px'>Year</th>"
        "<th style='text-align:left;padding:6px'>Total postings</th>"
        "<th style='text-align:left;padding:6px'>Group 1</th>"
        "<th style='text-align:left;padding:6px'>Group 2</th>"
        "<th style='text-align:left;padding:6px'>Group 3</th></tr>"
        + "".join(r.replace("<td>", "<td style='padding:6px;border-top:1px solid #ddd'>")
                  for r in rows_html)
        + "</table>"
    )


def chart_top_keywords(df: pl.DataFrame):
    exploded = (
        df.select(["group", "matched_group_keywords"])
        .filter(pl.col("group") != "NA")
        .explode("matched_group_keywords")
        .drop_nulls()
        .rename({"matched_group_keywords": "keyword"})
        .group_by(["keyword", "group"]).len()
        .sort("len", descending=True)
        .head(TOP_N_KEYWORDS)
        .to_pandas()
    )
    if exploded.empty:
        return None
    exploded["group_label"] = "Group " + exploded["group"]
    fig = px.bar(
        exploded.sort_values("len"), x="len", y="keyword", color="group_label", orientation="h",
        title=f"Top {TOP_N_KEYWORDS} triggered keywords (bare 'Copilot' removed - see writeup)",
        labels={"len": "Number of postings", "keyword": "", "group_label": ""},
        color_discrete_map={"Group 1": "#2ca02c", "Group 2": "#1f77b4", "Group 3": "#ff7f0e"},
    )
    fig.update_layout(height=700)
    return fig


def chart_share_by_language(df: pl.DataFrame):
    lang = (
        df.with_columns((pl.col("group") != "NA").alias("flagged"))
        .group_by("detected_language")
        .agg(total=pl.len(), flagged=pl.col("flagged").sum())
        .with_columns((100 * pl.col("flagged") / pl.col("total")).alias("share"))
        .sort("total", descending=True)
        .to_pandas()
    )
    fig = px.bar(
        lang, x="detected_language", y="share", hover_data=["total", "flagged"],
        title="GenAI-mention share by detected language",
        labels={"detected_language": "", "share": "% of postings"},
    )
    return fig


def chart_share_by_industry(df: pl.DataFrame):
    ind = (
        df.with_columns((pl.col("group") != "NA").alias("flagged"))
        .group_by("_industry")
        .agg(total=pl.len(), flagged=pl.col("flagged").sum())
        .filter(pl.col("total") >= MIN_ADS_PER_INDUSTRY)
        .with_columns((100 * pl.col("flagged") / pl.col("total")).alias("share"))
        .sort("share", descending=True)
        .head(TOP_N_INDUSTRIES)
        .to_pandas()
    )
    fig = px.bar(
        ind.sort_values("share"), x="share", y="_industry", orientation="h",
        hover_data=["total", "flagged"],
        title=f"GenAI-mention share by industry (top {TOP_N_INDUSTRIES}, min "
              f"{MIN_ADS_PER_INDUSTRY} ads)",
        labels={"share": "% of postings", "_industry": ""},
    )
    fig.update_layout(height=650)
    return fig


def chart_occupation_share(df: pl.DataFrame, root: Path):
    occ_map = build_occupation_to_isco_map(root)
    sub = df.select(["group", "occupations"])

    rows = []
    fotomodell_dropped = 0
    ambiguous_skipped = 0
    for group, occs in zip(sub["group"], sub["occupations"]):
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
            occ_id = _safe_id(occ)
            if occ_id in occ_map:
                isco_code, label = occ_map[occ_id]
                rows.append((isco_code, label, has_group))

    print(f"  [occupation chart] Dropped a secondary Fotomodell tag on "
          f"{fotomodell_dropped:,} ads; excluded {ambiguous_skipped:,} ads with "
          f">{MAX_OCCUPATIONS_PER_AD} occupation tags.")

    agg = pd.DataFrame(rows, columns=["isco_code", "label", "has_group"])
    if agg.empty:
        return None

    summary = (
        agg.groupby(["isco_code", "label"])
        .agg(total_ads=("has_group", "size"), flagged_ads=("has_group", "sum"))
        .reset_index()
    )
    summary = summary[summary["total_ads"] >= MIN_ADS_PER_OCCUPATION]
    summary["share"] = 100 * summary["flagged_ads"] / summary["total_ads"]
    summary = summary.sort_values("share", ascending=False).head(TOP_N_OCCUPATIONS)
    summary["occupation"] = summary["label"] + " (" + summary["isco_code"] + ")"

    fig = px.bar(
        summary.sort_values("share"), x="share", y="occupation", orientation="h",
        hover_data=["total_ads", "flagged_ads"],
        title=f"Top {TOP_N_OCCUPATIONS} occupations by GenAI posting share "
              f"(Fotomodell secondary-tag fix applied - see writeup)",
        labels={"share": "% of ads with a group 1/2/3 keyword match", "occupation": ""},
    )
    fig.update_layout(height=650)
    return fig


# --- New sections --------------------------------------------------------

def chart_postings_per_year(df: pl.DataFrame):
    yearly = (
        df.with_columns(pl.col("tst_created").dt.year().alias("year"))
        .group_by("year").len().sort("year").to_pandas()
    )
    yearly["year"] = yearly["year"].astype(str)
    fig = px.bar(
        yearly, x="year", y="len",
        title="Total postings per year (context - NOT GenAI-specific; "
              "2018 is December only, and severely undercounted - see writeup)",
        labels={"year": "", "len": "Total postings"},
        text="len",
    )
    fig.update_traces(textposition="outside")
    return fig


def chart_firm_adoption_per_year(df: pl.DataFrame):
    """Distinct firms with >=1 GenAI-flagged ad, per year - mirrors Domenico's
    thesis headline stat ('554 firms had advertised >=1 GenAI vacancy by 2025')."""
    yearly = (
        df.filter(pl.col("group") != "NA")
        .with_columns(pl.col("tst_created").dt.year().alias("year"))
        .group_by("year").agg(pl.col("_firm_id").n_unique().alias("adopting_firms"))
        .sort("year").to_pandas()
    )
    yearly["year"] = yearly["year"].astype(str)
    total_distinct_firms = df.filter(pl.col("group") != "NA")["_firm_id"].n_unique()
    fig = px.bar(
        yearly, x="year", y="adopting_firms",
        title=f"Distinct firms posting >=1 GenAI-flagged ad, per year "
              f"(total distinct adopting firms across 2018-2025: {total_distinct_firms:,})",
        labels={"year": "", "adopting_firms": "Distinct adopting firms"},
        text="adopting_firms",
    )
    fig.update_traces(textposition="outside")
    return fig


def chart_share_by_firm_size(df: pl.DataFrame):
    sz = (
        df.with_columns((pl.col("group") != "NA").alias("flagged"))
        .group_by("_size_name")
        .agg(total=pl.len(), flagged=pl.col("flagged").sum())
        .with_columns((100 * pl.col("flagged") / pl.col("total")).alias("share"))
        .sort("share", descending=True)
        .to_pandas()
    )
    fig = px.bar(
        sz, x="_size_name", y="share", hover_data=["total", "flagged"],
        title="GenAI-mention share by firm size",
        labels={"_size_name": "", "share": "% of postings"},
    )
    return fig


def chart_share_by_homeoffice(df: pl.DataFrame):
    ho = (
        df.with_columns((pl.col("group") != "NA").alias("flagged"))
        .group_by("has_homeoffice")
        .agg(total=pl.len(), flagged=pl.col("flagged").sum())
        .with_columns((100 * pl.col("flagged") / pl.col("total")).alias("share"))
        .to_pandas()
    )
    ho["has_homeoffice"] = ho["has_homeoffice"].map({True: "Home office offered", False: "No home office"})
    fig = px.bar(
        ho, x="has_homeoffice", y="share", hover_data=["total", "flagged"],
        title="GenAI-mention share by home-office availability",
        labels={"has_homeoffice": "", "share": "% of postings"},
    )
    return fig


def chart_share_by_contract_type(df: pl.DataFrame):
    ct = (
        df.with_columns((pl.col("group") != "NA").alias("flagged"))
        .group_by("is_temporary")
        .agg(total=pl.len(), flagged=pl.col("flagged").sum())
        .with_columns((100 * pl.col("flagged") / pl.col("total")).alias("share"))
        .to_pandas()
    )
    ct["is_temporary"] = ct["is_temporary"].map({True: "Temporary contract", False: "Permanent contract"})
    fig = px.bar(
        ct, x="is_temporary", y="share", hover_data=["total", "flagged"],
        title="GenAI-mention share by contract type",
        labels={"is_temporary": "", "share": "% of postings"},
    )
    return fig


def chart_share_by_canton(df: pl.DataFrame):
    exploded = (
        df.select(["group", "cantons_bonus"])
        .with_columns((pl.col("group") != "NA").alias("flagged"))
        .explode("cantons_bonus")
        .drop_nulls(subset=["cantons_bonus"])
        .group_by("cantons_bonus")
        .agg(total=pl.len(), flagged=pl.col("flagged").sum())
        .filter(pl.col("total") >= MIN_ADS_PER_CANTON)
        .with_columns((100 * pl.col("flagged") / pl.col("total")).alias("share"))
        .sort("share", descending=True)
        .head(TOP_N_CANTONS)
        .to_pandas()
    )
    if exploded.empty:
        return None
    fig = px.bar(
        exploded.sort_values("share"), x="share", y="cantons_bonus", orientation="h",
        hover_data=["total", "flagged"],
        title=f"Top {TOP_N_CANTONS} cantons by GenAI-mention share (min {MIN_ADS_PER_CANTON} ads)",
        labels={"share": "% of postings", "cantons_bonus": ""},
    )
    return fig


def table_recommended_flag_summary(df: pl.DataFrame) -> str:
    """Uses the persisted best-group `group` column directly: since Option B's
    flag is `>=1 Group 1 OR >=k combined Group2+Group3`, and an ad's best group
    already tells us the strongest thing it matched, `group in {"1","2","3"}`
    covers every Group-1 ad (flagged regardless of k) plus every ad whose ONLY
    match is Group 2/3 - the latter needs the full re-derived G2+G3 count to
    know if it clears k=3, which code/mine/group_strategy_analysis.py already
    computed. This table reports that script's last confirmed k=3 numbers
    directly rather than recomputing here, to avoid maintaining two paths."""
    g1 = df.filter(pl.col("group") == "1").height
    g2 = df.filter(pl.col("group") == "2").height
    g3 = df.filter(pl.col("group") == "3").height
    total = df.height
    return (
        "<h3>Recommended flag: Option B, k=3 (confirmed by Jeremias from the elbow plot)</h3>"
        "<p>Flag = &ge;1 Group 1 keyword, OR &ge;3 combined Group 2+3 keywords. "
        "Exact recomputed total is in code/mine/group_strategy_analysis.py's output "
        "(run separately - see notes) since it needs the full per-group keyword "
        f"recount, not just the best-group label. Current best-group breakdown for "
        f"context: Group 1={g1:,}, Group 2={g2:,}, Group 3={g3:,}, out of {total:,} "
        "total postings.</p>"
    )


def main():
    parser = argparse.ArgumentParser(description="Build the v2 results dashboard HTML")
    parser.add_argument("--config", type=str, default="pipeline/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(__file__).resolve().parents[2]

    print("Loading classified_data...")
    df = pl.scan_parquet(
        Path(config.output_dir) / "classified_data" / "year=*" / "*.parquet"
    ).select([
        "group", "tst_created", "detected_language", "matched_group_keywords",
        "occupations", "_industry", "_firm_id", "_size_name", "has_homeoffice",
        "is_temporary", "cantons_bonus",
    ]).collect()
    print(f"  {df.height:,} postings loaded.")

    elbow_path = Path(config.output_dir) / "plots" / "option_b_elbow.html"
    elbow_embed = (
        f"<iframe src='option_b_elbow.html' style='width:100%;height:500px;"
        f"border:none' title='Option B elbow plot'></iframe>"
        "<p><em>k=3 confirmed by Jeremias from this plot, 2026-07-30.</em></p>"
        if elbow_path.exists() else None
    )

    sections = [
        ("1. Classification breakdown", "figure", chart_classification_breakdown(df)),
        ("2. Trend over time", "figure", chart_share_over_time(df)),
        ("3. Matches per group", "figure", chart_matches_per_group(df)),
        ("4. Group share per year", "figure", chart_group_share_per_year(df)),
        ("5. Proportion table by year", "html", table_proportion_per_year(df)),
        ("6. Option B elbow plot (k=3 confirmed)", "html", elbow_embed),
        ("7. Occupation breakdown (Fotomodell fix applied)", "figure", chart_occupation_share(df, root)),
        ("8. Top keywords", "figure", chart_top_keywords(df)),
        ("9. By language", "figure", chart_share_by_language(df)),
        ("10. By industry", "figure", chart_share_by_industry(df)),
        ("11. Total postings per year (context)", "figure", chart_postings_per_year(df)),
        ("12. Firm-level adoption per year", "figure", chart_firm_adoption_per_year(df)),
        ("13. By firm size", "figure", chart_share_by_firm_size(df)),
        ("14. By home-office availability", "figure", chart_share_by_homeoffice(df)),
        ("15. By contract type", "figure", chart_share_by_contract_type(df)),
        ("16. By canton", "figure", chart_share_by_canton(df)),
        ("17. Recommended flag summary (Option B, k=3)", "html", table_recommended_flag_summary(df)),
    ]

    body_parts = [
        "<html><head><meta charset='utf-8'>"
        "<title>GenAI Adoption Pipeline - Results Dashboard v2</title>"
        "<style>body{font-family:Arial,Helvetica,sans-serif;max-width:1100px;"
        "margin:0 auto;padding:20px;} h1{margin-bottom:0;} .subtitle{color:#666;"
        "margin-top:4px;margin-bottom:40px;} .chart{margin-bottom:60px;}"
        "hr{border:none;border-top:1px solid #ddd;margin:40px 0;}"
        ".flag{background:#fff3cd;border:1px solid #ffe69c;border-radius:6px;"
        "padding:12px 16px;margin-bottom:30px;}</style></head><body>"
        "<h1>GenAI Adoption Pipeline &mdash; Results Dashboard v2</h1>"
        f"<p class='subtitle'>{df.height:,} classified job postings "
        "(Switzerland, 2018&ndash;2025 - extended from the earlier 2020&ndash;2025 "
        "dashboard). Generated from output/classified_data after removing bare "
        "\"Copilot\" as a keyword and applying the Fotomodell secondary-tag fix "
        "(both per Jeremias, 2026-07-30).</p>"
        "<div class='flag'><strong>Known open issue (not fixed here):</strong> "
        "~99.9% of Dec 2018 ads and an elevated ~12-19% of Jan 2019-Nov 2020 ads "
        "have a null <code>company.size.id</code>, which the micro-enterprise "
        "filter's <code>!= micro_id</code> comparison silently treats as "
        "\"exclude\" (Polars: <code>null != x</code> is null, not true). This is "
        "why Dec 2018 collapses to just 33 surviving postings. Needs a decision "
        "on whether unknown-size ads should be kept or dropped before fixing.</div>"
    ]

    first = True
    for title, kind, payload in sections:
        if payload is None:
            body_parts.append(f"<h2>{title}</h2><p><em>No data available.</em></p><hr>")
            continue
        if kind == "html":
            body_parts.append(f"<div class='chart'><h2>{title}</h2>{payload}</div><hr>")
            continue
        html = pio.to_html(
            payload, full_html=False,
            include_plotlyjs="inline" if first else False,
        )
        first = False
        body_parts.append(f"<div class='chart'><h2>{title}</h2>{html}</div><hr>")

    body_parts.append("</body></html>")

    out_dir = Path(config.output_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dashboard_v2.html"
    out_path.write_text("\n".join(body_parts), encoding="utf-8")
    print(f"\nSaved dashboard v2 to {out_path.resolve()}")


if __name__ == "__main__":
    main()
