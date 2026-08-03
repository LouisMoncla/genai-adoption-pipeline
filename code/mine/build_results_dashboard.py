"""
Single-file HTML dashboard summarizing pipeline/group_classification.py's output
(output/classified_data/year=*/*.parquet), so the results can be browsed by just
double-clicking one file - no server, no notebook, no plotly/polars install needed
on the viewing machine.

Ten sections, in order:
  1. Overall classification breakdown (group 1/2/3/NA counts)
  2. GenAI-mention share of postings over time (monthly) - the headline trend chart
  3. Matches per group (plain G1/G2/G3 count bar, requested explicitly alongside the
     breakdown pie in (1))
  4. Group share per year (per-year, per-group breakdown - distinct from (2)'s
     monthly-overall-share line)
  5. Proportion table: matches per group per year (a literal cross-tab, not a chart)
  6. Option B elbow plot (k vs. total flagged) - embedded from
     output/plots/option_b_elbow.html (code/mine/group_strategy_analysis.py), via
     iframe rather than recomputed here, since that script already owns this
     computation
  7. Top 20 occupations by GenAI posting share (reuses the ISCO mapping logic from
     occupation_genai_share_plot.py, including its MAX_JOBS>5 ambiguous-ad exclusion)
  8. Top 25 triggered keywords, colored by group
  9. GenAI share by detected language
  10. GenAI share by industry (top 20 by posting volume)

Each chart is an interactive Plotly figure. plotly.js is embedded once (inline, not
CDN) so the file works fully offline.
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
TOP_N_OCCUPATIONS = 20
TOP_N_KEYWORDS = 25
TOP_N_INDUSTRIES = 20
# Matches code/mine/occupation_genai_share_plot.py's MAX_OCCUPATIONS_PER_AD - see that
# file's comment (mirrors Jeremias's own R scripts' MAX_JOBS <- 5L ambiguous-ad cutoff).
MAX_OCCUPATIONS_PER_AD = 5


def _load_isco_titles(root: Path) -> dict:
    structure = pd.read_csv(root / "data" / "raw" / "isco_structure.csv", encoding="cp1252")
    structure = structure[structure["ISCO_version"] == "ISCO-08"]
    return dict(zip(structure["unit"].astype(int), structure["description"].str.strip()))


def build_occupation_to_isco_map(root: Path) -> dict:
    """Returns {x28_occupation_id: (isco_code, display_label)}. Same logic as
    occupation_genai_share_plot.py - duplicated rather than imported so this
    dashboard has no dependency on that script's CLI/argparse setup."""
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


def chart_classification_breakdown(df: pl.DataFrame) -> "px.Figure":
    counts = df.group_by("group").len().sort("group").to_pandas()
    counts["label"] = counts["group"].map({
        "1": "Group 1 (core GenAI)", "2": "Group 2", "3": "Group 3", "NA": "No keyword match",
    })
    fig = px.pie(
        counts, names="label", values="len", hole=0.45,
        title="Overall classification breakdown (all postings)",
        color="label",
        color_discrete_map={
            "Group 1 (core GenAI)": "#2ca02c", "Group 2": "#1f77b4",
            "Group 3": "#ff7f0e", "No keyword match": "#d3d3d3",
        },
    )
    fig.update_traces(textinfo="label+percent+value")
    return fig


def chart_share_over_time(df: pl.DataFrame) -> "px.Figure":
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
        title="GenAI-mention share of job postings over time",
        labels={"month": "", "share_pct": "% of postings", "series": ""},
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def chart_matches_per_group(df: pl.DataFrame) -> "px.Figure":
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
    """Shared by the per-year chart and the proportion table below - one
    (year, group) -> count pivot, with every group column present even if a
    year happens to have zero ads in it."""
    yearly = (
        df.with_columns(pl.col("tst_created").dt.year().alias("year"))
        .group_by(["year", "group"]).len()
    )
    pivot = yearly.pivot(on="group", index="year", values="len").fill_null(0).sort("year")
    for g in ("1", "2", "3", "NA"):
        if g not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0).alias(g))
    return pivot


def chart_group_share_per_year(df: pl.DataFrame) -> "px.Figure":
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
        title="Group share per year (each group's % of that year's postings)",
        labels={"year": "", "share_pct": "% of that year's postings", "group": ""},
        color_discrete_map={"Group 1": "#2ca02c", "Group 2": "#1f77b4", "Group 3": "#ff7f0e"},
    )
    return fig


def table_proportion_per_year(df: pl.DataFrame) -> str:
    """Literal cross-tab (not a chart): matches per group per year, as an HTML
    table - the count and its share of that year's total postings side by side."""
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


def chart_occupation_share(df: pl.DataFrame, root: Path) -> "px.Figure | None":
    occ_map = build_occupation_to_isco_map(root)
    sub = df.select(["group", "occupations"])

    rows = []
    for group, occs in zip(sub["group"], sub["occupations"]):
        has_group = group != "NA"
        if occs is None or len(occs) > MAX_OCCUPATIONS_PER_AD:
            continue
        for occ in occs:
            try:
                occ_id = int(occ["id"])
            except (TypeError, ValueError):
                continue
            if occ_id in occ_map:
                isco_code, label = occ_map[occ_id]
                rows.append((isco_code, label, has_group))

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
              f"(min {MIN_ADS_PER_OCCUPATION} ads; x28 occupation labels are known to be "
              "noisy for some titles - see notes/meeting_2026-07-24_pipeline_run_report.md)",
        labels={"share": "% of ads with a group 1/2/3 keyword match", "occupation": ""},
    )
    fig.update_layout(height=650)
    return fig


def chart_top_keywords(df: pl.DataFrame) -> "px.Figure | None":
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
        title=f"Top {TOP_N_KEYWORDS} triggered keywords",
        labels={"len": "Number of postings", "keyword": "", "group_label": ""},
        color_discrete_map={"Group 1": "#2ca02c", "Group 2": "#1f77b4", "Group 3": "#ff7f0e"},
    )
    fig.update_layout(height=700)
    return fig


def chart_share_by_language(df: pl.DataFrame) -> "px.Figure":
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


def chart_share_by_industry(df: pl.DataFrame) -> "px.Figure":
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


def main():
    parser = argparse.ArgumentParser(description="Build the results dashboard HTML")
    parser.add_argument("--config", type=str, default="pipeline/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(__file__).resolve().parents[2]

    print("Loading classified_data...")
    df = pl.scan_parquet(
        Path(config.output_dir) / "classified_data" / "year=*" / "*.parquet"
    ).select([
        "group", "tst_created", "detected_language", "matched_group_keywords",
        "occupations", "_industry",
    ]).collect()
    print(f"  {df.height:,} postings loaded.")

    elbow_path = Path(config.output_dir) / "plots" / "option_b_elbow.html"
    elbow_embed = (
        f"<iframe src='option_b_elbow.html' style='width:100%;height:500px;"
        f"border:none' title='Option B elbow plot'></iframe>"
        if elbow_path.exists() else None
    )

    sections = [
        ("1. Classification breakdown", "figure", chart_classification_breakdown(df)),
        ("2. Trend over time", "figure", chart_share_over_time(df)),
        ("3. Matches per group", "figure", chart_matches_per_group(df)),
        ("4. Group share per year", "figure", chart_group_share_per_year(df)),
        ("5. Proportion table by year", "html", table_proportion_per_year(df)),
        ("6. Option B elbow plot (k vs. total flagged)", "html", elbow_embed),
        ("7. Occupation breakdown", "figure", chart_occupation_share(df, root)),
        ("8. Top keywords", "figure", chart_top_keywords(df)),
        ("9. By language", "figure", chart_share_by_language(df)),
        ("10. By industry", "figure", chart_share_by_industry(df)),
    ]

    body_parts = [
        "<html><head><meta charset='utf-8'>"
        "<title>GenAI Adoption Pipeline - Results Dashboard</title>"
        "<style>body{font-family:Arial,Helvetica,sans-serif;max-width:1100px;"
        "margin:0 auto;padding:20px;} h1{margin-bottom:0;} .subtitle{color:#666;"
        "margin-top:4px;margin-bottom:40px;} .chart{margin-bottom:60px;}"
        "hr{border:none;border-top:1px solid #ddd;margin:40px 0;}</style></head><body>"
        "<h1>GenAI Adoption Pipeline &mdash; Results Dashboard</h1>"
        f"<p class='subtitle'>{df.height:,} classified job postings "
        "(Switzerland, 2020&ndash;2025). Generated from output/classified_data. "
        "See notes/meeting_2026-07-24_pipeline_run_report.md for full methodology, "
        "known limitations, and confidence levels per result.</p>"
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
        body_parts.append(f"<div class='chart'>{html}</div><hr>")

    body_parts.append("</body></html>")

    out_dir = Path(config.output_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dashboard.html"
    out_path.write_text("\n".join(body_parts), encoding="utf-8")
    print(f"\nSaved dashboard to {out_path.resolve()}")


if __name__ == "__main__":
    main()
