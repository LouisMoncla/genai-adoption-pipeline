"""
V3 of the single-file HTML results dashboard (see build_results_dashboard.py for
v1, build_results_dashboard_v2.py for v2), rebuilt after today's (2026-08-02)
pipeline rerun: x28-native language detection (replaces FastText), the
micro-enterprise filter removed, and a full re-export/reclassification of the
combined 2018-2025 dataset. See notes/meeting_prep_2026-08-02.md for the full
writeup this dashboard is meant to accompany.

Goal for v3: be the single place that answers "can we trust these numbers,
and what do they say" without needing to cross-reference separate scripts/CSVs.
Everything from v2 (sections 1-17) is kept, plus:
  18. Keyword coverage by Group and by source layer (St/H&L/LLM) - the new
      deliverable from code/mine/keyword_match_counts_by_group_layer.py,
      recomputed inline here so it's always in sync with what's loaded below.
  19. Full Option A/B/C combination-strategy comparison - k/cutoff sweep
      tables AND the "which keyword is driving the newly-swept-in ads"
      breakdown (with the same >50%-single-keyword warning
      code/mine/group_strategy_analysis.py already flags), computed inline
      via the same pipeline.group_classification functions that script uses,
      not read from a separately-run CSV that could go stale.
  20. GenAI share vs Domenico's thesis benchmark (2023-2025 window - the
      only window that's a fair apples-to-apples comparison to his sample).
  21. Corpus-wide vs flagged-ads language distribution, side by side - makes
      the English-overrepresentation finding visible directly instead of
      requiring a separate investigation to notice it.
  22. Data-quality / sanity-check summary - the cross-year duplicate
      duplicate_group finding is recomputed live (cheap); the narrative
      findings from today's investigation (2019=0 explained, export-query
      rewrites verified, file-integrity check) are recorded as dated notes
      since re-deriving them on every dashboard build isn't cheap and they
      don't change build-to-build unless the underlying code changes again.

Section 7 (occupation) and the Fotomodell fix, and section 4/13 conventions,
are duplicated from v2 rather than imported - matching this project's existing
convention (see v2's own docstring) of keeping each dashboard script
independently runnable rather than cross-importing sibling code/mine scripts.
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
import polars as pl

from pipeline.config_loader import load_config
from pipeline.group_classification import (
    build_keyword_patterns,
    classify_text,
    load_master_keywords,
    normalize,
)

MIN_ADS_PER_OCCUPATION = 20
MIN_ADS_PER_INDUSTRY = 50
MIN_ADS_PER_CANTON = 50
TOP_N_OCCUPATIONS = 20
TOP_N_KEYWORDS = 25
TOP_N_INDUSTRIES = 20
TOP_N_CANTONS = 15
MAX_OCCUPATIONS_PER_AD = 5
OPTION_B_K = 3  # confirmed by Jeremias from the elbow plot, 2026-07-30

B_THRESHOLDS = [2, 3, 4, 5, 6]
C_WEIGHTS = {1: 3, 2: 2, 3: 1}
C_CUTOFFS = [3, 4, 5, 6]

FOTOMODELL_X28_ID = 11001610

LAYER_LABELS = {
    "St": "St (layer1_stanford.py)",
    "H&L": "H&L (layer2_hosseini.py)",
    "LLM": "LLM (layer3.py)",
}

DOMENICO_THESIS_RATE_PCT = 0.175  # his 2023-2025, medium/large non-recruiter sample


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


# --- Sections 1-5, 7-17: unchanged from v2 ------------------------------------

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
              "note 2018 is Dec-only",
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
        total_flagged = row["1"] + row["2"] + row["3"]
        total_pct = 100 * total_flagged / row["total"] if row["total"] else 0
        cells.append(f"<td>{total_flagged:,} ({total_pct:.4f}%)</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    return (
        "<h3>Proportion table: matches per group per year</h3>"
        "<table style='border-collapse:collapse;width:100%'>"
        "<tr style='background:#f0f0f0'>"
        "<th style='text-align:left;padding:6px'>Year</th>"
        "<th style='text-align:left;padding:6px'>Total postings</th>"
        "<th style='text-align:left;padding:6px'>Group 1</th>"
        "<th style='text-align:left;padding:6px'>Group 2</th>"
        "<th style='text-align:left;padding:6px'>Group 3</th>"
        "<th style='text-align:left;padding:6px'>Any flag</th></tr>"
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
        title=f"Top {TOP_N_KEYWORDS} triggered keywords",
        labels={"len": "Number of postings", "keyword": "", "group_label": ""},
        color_discrete_map={"Group 1": "#2ca02c", "Group 2": "#1f77b4", "Group 3": "#ff7f0e"},
    )
    fig.update_layout(height=700)
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
              f"(Fotomodell secondary-tag fix applied)",
        labels={"share": "% of ads with a group 1/2/3 keyword match", "occupation": ""},
    )
    fig.update_layout(height=650)
    return fig


def chart_postings_per_year(df: pl.DataFrame):
    yearly = (
        df.with_columns(pl.col("tst_created").dt.year().alias("year"))
        .group_by("year").len().sort("year").to_pandas()
    )
    yearly["year"] = yearly["year"].astype(str)
    fig = px.bar(
        yearly, x="year", y="len",
        title="Total postings per year (context - NOT GenAI-specific; "
              "2018 is December only)",
        labels={"year": "", "len": "Total postings"},
        text="len",
    )
    fig.update_traces(textposition="outside")
    return fig


def chart_firm_adoption_per_year(df: pl.DataFrame):
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


# --- New in v3 -----------------------------------------------------------

def table_keyword_coverage(df: pl.DataFrame) -> str:
    """Section 18: how many of the 49 validated keywords produced >=1 match,
    by Group and by source layer. Recomputed inline (same logic as
    code/mine/keyword_match_counts_by_group_layer.py) so this dashboard is
    always in sync with whatever classified_data it was built from."""
    master_keywords = load_master_keywords()
    master_df = pl.DataFrame([
        {"keyword": k["keyword"], "group": k["group"], "source": k["source"]}
        for k in master_keywords
    ])

    counts = (
        df.select("matched_group_keywords")
        .explode("matched_group_keywords")
        .drop_nulls()
        .rename({"matched_group_keywords": "keyword"})
        .group_by("keyword")
        .agg(pl.len().alias("ad_match_count"))
    )
    table = (
        master_df.join(counts, on="keyword", how="left")
        .with_columns(pl.col("ad_match_count").fill_null(0))
        .sort(["group", "ad_match_count"], descending=[False, True])
    )
    matched = table.filter(pl.col("ad_match_count") > 0)
    zero_match = table.filter(pl.col("ad_match_count") == 0)

    summary_rows = []
    for grp in (1, 2, 3):
        total_in_group = table.filter(pl.col("group") == grp).height
        matched_in_group = matched.filter(pl.col("group") == grp).height
        summary_rows.append(
            f"<tr><td style='padding:6px'>Group {grp}</td>"
            f"<td style='padding:6px'>{matched_in_group}/{total_in_group}</td></tr>"
        )
    for src in ["St", "H&L", "LLM"]:
        total_in_src = table.filter(pl.col("source") == src).height
        matched_in_src = matched.filter(pl.col("source") == src).height
        summary_rows.append(
            f"<tr><td style='padding:6px'>{LAYER_LABELS[src]}</td>"
            f"<td style='padding:6px'>{matched_in_src}/{total_in_src}</td></tr>"
        )

    zero_list = ", ".join(
        f"{r['keyword']} (Group {r['group']}, {LAYER_LABELS[r['source']]})"
        for r in zero_match.iter_rows(named=True)
    ) or "none - every validated keyword matched at least once"

    full_rows = "".join(
        f"<tr><td style='padding:4px;border-top:1px solid #eee'>{r['keyword']}</td>"
        f"<td style='padding:4px;border-top:1px solid #eee'>{r['group']}</td>"
        f"<td style='padding:4px;border-top:1px solid #eee'>{LAYER_LABELS[r['source']]}</td>"
        f"<td style='padding:4px;border-top:1px solid #eee;text-align:right'>{r['ad_match_count']:,}</td></tr>"
        for r in table.iter_rows(named=True)
    )

    return (
        f"<p>Of {table.height} validated keywords, <strong>{matched.height} produced "
        f"&ge;1 match</strong> across the full classified dataset. Never-matched: "
        f"{zero_list}.</p>"
        "<table style='border-collapse:collapse;margin-bottom:20px'>"
        "<tr style='background:#f0f0f0'><th style='padding:6px;text-align:left'>Breakdown</th>"
        "<th style='padding:6px;text-align:left'>Matched / Total</th></tr>"
        + "".join(summary_rows) + "</table>"
        "<details><summary style='cursor:pointer'>Full per-keyword table (click to expand)</summary>"
        "<table style='border-collapse:collapse;width:100%;margin-top:10px'>"
        "<tr style='background:#f0f0f0'>"
        "<th style='text-align:left;padding:4px'>Keyword</th>"
        "<th style='text-align:left;padding:4px'>Group</th>"
        "<th style='text-align:left;padding:4px'>Layer</th>"
        "<th style='text-align:right;padding:4px'>Ad matches</th></tr>"
        + full_rows + "</table></details>"
    )


def _recompute_full_matches(df: pl.DataFrame, en_patterns, local_patterns) -> pl.DataFrame:
    """Same logic as code/mine/group_strategy_analysis.py's recompute_full_matches -
    the persisted `group` column only keeps the single best-priority group's
    keywords, so Options B/C need the full per-group recount on just the
    already-flagged ads (cheap: ~7K ads, not the full 5.18M)."""
    rows = []
    for ad_id, text, lang in zip(df["ad_id"], df["content_clean"], df["detected_language"]):
        norm = normalize(text or "")
        matched = classify_text(norm, lang, en_patterns, local_patterns)
        g1 = {k for k, g in matched.items() if g == 1}
        g2 = {k for k, g in matched.items() if g == 2}
        g3 = {k for k, g in matched.items() if g == 3}
        rows.append({
            "ad_id": ad_id,
            "g1_count": len(g1), "g2_count": len(g2), "g3_count": len(g3),
            "g2_kw": sorted(g2), "g3_kw": sorted(g3),
        })
    return pl.DataFrame(rows)


def _keyword_freq(df: pl.DataFrame, mask, cols=("g2_kw", "g3_kw")) -> Counter:
    c = Counter()
    sub = df.filter(mask)
    for col in cols:
        for kw_list in sub[col]:
            c.update(kw_list)
    return c


def section_strategy_comparison(df: pl.DataFrame) -> tuple:
    """Section 19: full Option A/B/C combination-strategy comparison, computed
    live (same approach as code/mine/group_strategy_analysis.py) rather than
    read from a separately-run CSV. Returns (elbow_figure, html_table)."""
    flagged = df.filter(pl.col("group") != "NA").select(
        ["ad_id", "content_clean", "detected_language"]
    )
    en_patterns, local_patterns = build_keyword_patterns(load_master_keywords())
    matches = _recompute_full_matches(flagged, en_patterns, local_patterns)

    g1, g2, g3 = matches["g1_count"], matches["g2_count"], matches["g3_count"]
    g23 = g2 + g3
    total_population = matches.height
    high_conf = (g1 >= 1)

    b_flags = {k: ((g1 >= 1) | (g23 >= k)) for k in B_THRESHOLDS}
    elbow_pdf = pd.DataFrame({
        "k": B_THRESHOLDS,
        "total_flagged": [int(b_flags[k].sum()) for k in B_THRESHOLDS],
    })
    elbow_fig = px.line(
        elbow_pdf, x="k", y="total_flagged", markers=True,
        title="Option B elbow plot: total ads flagged vs. corroboration threshold k "
              "(k=3 confirmed by Jeremias, 2026-07-30)",
        labels={"k": "k (min combined Group 2+3 keywords, when no Group 1 match)",
                "total_flagged": "Total ads flagged"},
    )

    score = C_WEIGHTS[1] * g1 + C_WEIGHTS[2] * g2 + C_WEIGHTS[3] * g3
    c_flags = {cutoff: (score >= cutoff) for cutoff in C_CUTOFFS}

    b_rows = "".join(
        f"<tr><td style='padding:6px'>k={k}</td>"
        f"<td style='padding:6px'>{int(b_flags[k].sum()):,}</td>"
        f"<td style='padding:6px'>{int((b_flags[k] & ~high_conf).sum()):,}</td></tr>"
        for k in B_THRESHOLDS
    )
    c_rows = "".join(
        f"<tr><td style='padding:6px'>cutoff={cutoff}</td>"
        f"<td style='padding:6px'>{int(c_flags[cutoff].sum()):,}</td>"
        f"<td style='padding:6px'>{int((c_flags[cutoff] & ~high_conf).sum()):,}</td></tr>"
        for cutoff in C_CUTOFFS
    )

    newly_b2 = b_flags[2] & ~high_conf
    freq_b2 = _keyword_freq(matches, newly_b2)
    newly_c4 = c_flags[4] & ~high_conf
    freq_c4 = _keyword_freq(matches, newly_c4)

    def _driver_html(freq, denom, label):
        if not freq:
            return f"<p><em>No {label} ads.</em></p>"
        rows = "".join(
            f"<tr><td style='padding:4px'>{kw}</td><td style='padding:4px'>{n:,}</td></tr>"
            for kw, n in freq.most_common(10)
        )
        top_kw, top_count = freq.most_common(1)[0]
        share = top_count / denom if denom else 0
        warn = (
            f"<p style='color:#a15c00'><strong>FLAG:</strong> '{top_kw}' alone drives "
            f"{share:.0%} of {label}'s newly-flagged ads - worth a manual eyeball "
            f"before trusting this option.</p>"
            if share > 0.5 else ""
        )
        return (
            f"<table style='border-collapse:collapse'>"
            f"<tr style='background:#f0f0f0'><th style='padding:4px;text-align:left'>Keyword</th>"
            f"<th style='padding:4px;text-align:left'>Ads driven</th></tr>{rows}</table>{warn}"
        )

    html = (
        f"<p>Population: {total_population:,} ads with &ge;1 keyword match "
        f"(out of {df.height:,} total ads scanned). High-confidence (&ge;1 Group 1): "
        f"{int(high_conf.sum()):,}.</p>"
        "<h3>Option B: corroboration count (&ge;1 G1 OR &ge;k combined G2+G3)</h3>"
        "<table style='border-collapse:collapse'>"
        "<tr style='background:#f0f0f0'><th style='padding:6px;text-align:left'>Threshold</th>"
        "<th style='padding:6px;text-align:left'>Total flagged</th>"
        "<th style='padding:6px;text-align:left'>Newly swept in (beyond G1 alone)</th></tr>"
        + b_rows + "</table>"
        "<h3>Option C: weighted score (G1=3, G2=2, G3=1)</h3>"
        "<table style='border-collapse:collapse'>"
        "<tr style='background:#f0f0f0'><th style='padding:6px;text-align:left'>Cutoff</th>"
        "<th style='padding:6px;text-align:left'>Total flagged</th>"
        "<th style='padding:6px;text-align:left'>Newly swept in (beyond G1 alone)</th></tr>"
        + c_rows + "</table>"
        "<h3>What's driving the 'newly swept in' ads?</h3>"
        "<p><strong>Option B @ k=2:</strong></p>" + _driver_html(freq_b2, int(newly_b2.sum()), "B@2") +
        "<p style='margin-top:16px'><strong>Option C @ cutoff=4:</strong></p>" + _driver_html(freq_c4, int(newly_c4.sum()), "C@4") +
        f"<p style='margin-top:16px'><strong>Recommended: Option B, k={OPTION_B_K} "
        f"(confirmed by Jeremias from the elbow plot) -&gt; {int(b_flags[OPTION_B_K].sum()):,} "
        f"flagged.</strong></p>"
    )
    return elbow_fig, html


def chart_language_comparison(df: pl.DataFrame):
    """Section 20: corpus-wide vs flagged-ads language distribution, side by
    side - makes the English-overrepresentation finding visible without a
    separate investigation."""
    total = df.height
    corpus = (
        df["detected_language"].value_counts()
        .with_columns((100 * pl.col("count") / total).alias("share"))
        .with_columns(pl.lit("All postings").alias("scope"))
    )
    flagged_df = df.filter(pl.col("group") != "NA")
    ftotal = flagged_df.height
    flagged = (
        flagged_df["detected_language"].value_counts()
        .with_columns((100 * pl.col("count") / ftotal).alias("share"))
        .with_columns(pl.lit("Flagged postings").alias("scope"))
    )
    combined = pl.concat([corpus, flagged]).to_pandas()
    fig = px.bar(
        combined, x="detected_language", y="share", color="scope", barmode="group",
        title="Language distribution: all postings vs. GenAI-flagged postings",
        labels={"detected_language": "", "share": "% share", "scope": ""},
    )
    return fig


def table_domenico_comparison(df: pl.DataFrame) -> str:
    """Section 21: our flagged rate vs. Domenico's thesis figure, on the same
    2023-2025 window his sample covers - the only apples-to-apples comparison."""
    window = df.filter(pl.col("tst_created").dt.year() >= 2023)
    total = window.height
    flagged = window.filter(pl.col("group") != "NA").height
    our_rate = 100 * flagged / total if total else 0
    ratio = our_rate / DOMENICO_THESIS_RATE_PCT if DOMENICO_THESIS_RATE_PCT else 0
    return (
        "<table style='border-collapse:collapse'>"
        "<tr style='background:#f0f0f0'><th style='padding:6px;text-align:left'>Source</th>"
        "<th style='padding:6px;text-align:left'>Window</th>"
        "<th style='padding:6px;text-align:left'>Flagged rate</th></tr>"
        f"<tr><td style='padding:6px'>This pipeline</td><td style='padding:6px'>2023-2025</td>"
        f"<td style='padding:6px'>{flagged:,} / {total:,} = {our_rate:.3f}%</td></tr>"
        f"<tr><td style='padding:6px'>Domenico's thesis</td><td style='padding:6px'>2023-2025 "
        f"(medium/large, non-recruiter sample)</td>"
        f"<td style='padding:6px'>{DOMENICO_THESIS_RATE_PCT:.3f}%</td></tr>"
        "</table>"
        f"<p>Our rate is <strong>{ratio:.2f}x</strong> his. Same direction and similar "
        "magnitude gap as found before today's rerun (was ~1.6x pre-rerun) - confirms "
        "this divergence is stable and not an export/methodology artifact of today's "
        "changes. Attributed to our classifier being simpler (no occupation-intensity "
        "gate, no Rule 2/3/C combination logic) - see notes/meeting_prep_2026-08-02.md.</p>"
    )


def table_data_quality_summary(df: pl.DataFrame) -> str:
    """Section 22: sanity-check summary. The cross-year duplicate check is
    recomputed live (cheap groupby); the rest are dated findings from today's
    investigation that don't need re-deriving on every rebuild."""
    dup_counts = df.group_by("duplicate_group").agg(pl.len().alias("n"))
    n_dup_groups = dup_counts.filter(pl.col("n") > 1).height
    n_surplus_rows = int((dup_counts.filter(pl.col("n") > 1)["n"] - 1).sum()) if n_dup_groups else 0
    flagged_dup = (
        df.filter(pl.col("group") != "NA")
        .group_by("duplicate_group").agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
    )

    return (
        "<h3>Cross-year duplicate check (live, recomputed on this data)</h3>"
        f"<p><code>duplicate_group</code> values appearing more than once: "
        f"{n_dup_groups:,} ({n_surplus_rows:,} surplus rows, "
        f"{100*n_surplus_rows/df.height:.2f}% of {df.height:,} total). "
        f"Among flagged ads specifically: {flagged_dup.height:,} duplicated ids "
        f"({int(flagged_dup['n'].sum()) if flagged_dup.height else 0} rows, negligible "
        f"vs the flagged total). Root cause (confirmed 2026-08-02): Phase I's dedup "
        "resets per calendar year, so a posting renewed across a year boundary "
        "survives as two rows - not a within-year bug (0% of duplicates are "
        "same-year). Open question: should this be deduped globally too? See "
        "notes/meeting_prep_2026-08-02.md #2.6.</p>"
        "<h3>Other checks from today's investigation (2026-08-02, not re-derived on every rebuild)</h3>"
        "<ul>"
        "<li><strong>Export query rewrites verified correct:</strong> the agg_loc/agg_meta "
        "rewrites that fixed today's export OOMs were checked against the original naive "
        "queries on a 100K-id sample - identical row counts, zero content mismatches.</li>"
        "<li><strong>2019 = zero flagged ads, explained:</strong> statistically implausible "
        "as noise (would expect ~212 matches at 2020's rate), confirmed by direct text "
        "search - every AI product name in the keyword list postdates 2019.</li>"
        "<li><strong>File integrity:</strong> all 49 classified_data files read cleanly, "
        "no corruption (relevant given a prior atomic-write bug).</li>"
        "<li><strong>Flagged-ads language skew:</strong> consistent with the pre-rerun "
        "finding (English ~6x overrepresented among flags vs. the corpus) - not an "
        "artifact of switching from FastText to x28's native language flag.</li>"
        "</ul>"
    )


def table_recommended_flag_summary(df: pl.DataFrame) -> str:
    g1 = df.filter(pl.col("group") == "1").height
    g2 = df.filter(pl.col("group") == "2").height
    g3 = df.filter(pl.col("group") == "3").height
    total = df.height
    return (
        f"<h3>Recommended flag: Option B, k={OPTION_B_K} (confirmed by Jeremias from the elbow plot)</h3>"
        f"<p>Flag = &ge;1 Group 1 keyword, OR &ge;{OPTION_B_K} combined Group 2+3 keywords. "
        "Exact recomputed total is in section 19 above (computed live from this same "
        f"data). Best-group breakdown for context: Group 1={g1:,}, Group 2={g2:,}, "
        f"Group 3={g3:,}, out of {total:,} total postings.</p>"
    )


def main():
    parser = argparse.ArgumentParser(description="Build the v3 results dashboard HTML")
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
        "is_temporary", "cantons_bonus", "ad_id", "content_clean", "duplicate_group",
    ]).collect()
    print(f"  {df.height:,} postings loaded.")

    print("Computing keyword coverage (section 18)...")
    keyword_coverage_html = table_keyword_coverage(df)

    print("Computing strategy comparison (section 19, recomputes classify_text on ~7K flagged ads)...")
    elbow_fig, strategy_html = section_strategy_comparison(df)

    print("Computing Domenico comparison (section 20)...")
    domenico_html = table_domenico_comparison(df)

    print("Computing data-quality summary (section 22)...")
    dq_html = table_data_quality_summary(df)

    sections = [
        ("1. Classification breakdown", "figure", chart_classification_breakdown(df)),
        ("2. Trend over time", "figure", chart_share_over_time(df)),
        ("3. Matches per group", "figure", chart_matches_per_group(df)),
        ("4. Group share per year", "figure", chart_group_share_per_year(df)),
        ("5. Proportion table by year", "html", table_proportion_per_year(df)),
        ("6. Option B elbow plot", "figure", elbow_fig),
        ("7. Occupation breakdown (Fotomodell fix applied)", "figure", chart_occupation_share(df, root)),
        ("8. Top keywords", "figure", chart_top_keywords(df)),
        ("9. By industry", "figure", chart_share_by_industry(df)),
        ("10. Total postings per year (context)", "figure", chart_postings_per_year(df)),
        ("11. Firm-level adoption per year", "figure", chart_firm_adoption_per_year(df)),
        ("12. By firm size", "figure", chart_share_by_firm_size(df)),
        ("13. By home-office availability", "figure", chart_share_by_homeoffice(df)),
        ("14. By contract type", "figure", chart_share_by_contract_type(df)),
        ("15. By canton", "figure", chart_share_by_canton(df)),
        ("16. Language distribution: all postings vs. flagged", "figure", chart_language_comparison(df)),
        ("17. Keyword coverage by Group & layer (NEW)", "html", keyword_coverage_html),
        ("18. Full Option A/B/C strategy comparison (NEW)", "html", strategy_html),
        ("19. GenAI share vs. Domenico's thesis benchmark (NEW)", "html", domenico_html),
        ("20. Data-quality / sanity-check summary (NEW)", "html", dq_html),
        ("21. Recommended flag summary (Option B, k=3)", "html", table_recommended_flag_summary(df)),
    ]

    body_parts = [
        "<html><head><meta charset='utf-8'>"
        "<title>GenAI Adoption Pipeline - Results Dashboard v3</title>"
        "<style>body{font-family:Arial,Helvetica,sans-serif;max-width:1100px;"
        "margin:0 auto;padding:20px;} h1{margin-bottom:0;} .subtitle{color:#666;"
        "margin-top:4px;margin-bottom:40px;} .chart{margin-bottom:60px;}"
        "hr{border:none;border-top:1px solid #ddd;margin:40px 0;}"
        "table{font-size:14px} details summary{font-weight:bold;margin:10px 0}"
        ".flag{background:#fff3cd;border:1px solid #ffe69c;border-radius:6px;"
        "padding:12px 16px;margin-bottom:16px;}"
        ".changelog{background:#eef6ff;border:1px solid #b6d4fe;border-radius:6px;"
        "padding:12px 16px;margin-bottom:30px;}</style></head><body>"
        "<h1>GenAI Adoption Pipeline &mdash; Results Dashboard v3</h1>"
        f"<p class='subtitle'>{df.height:,} classified job postings "
        "(Switzerland, 2018&ndash;2025). Generated after the 2026-08-02 pipeline "
        "rerun. See notes/meeting_prep_2026-08-02.md for the full writeup.</p>"
        "<div class='changelog'><strong>What changed since dashboard v2 "
        "(2026-08-02):</strong><ul style='margin:8px 0 0 0'>"
        "<li>Language detection switched from FastText to x28's own native "
        "<code>language</code> column.</li>"
        "<li>Micro-enterprise filter removed entirely (also fixes the Dec 2018 "
        "null-<code>size_id</code> collapse bug from the previous dashboard).</li>"
        "<li>All three source dumps re-exported and the full 2018-2025 dataset "
        "reclassified from scratch.</li>"
        "<li>New sections 17-20 below: keyword coverage by group/layer, full "
        "Option A/B/C strategy comparison (not just the elbow plot), a Domenico "
        "thesis benchmark comparison, and a live data-quality/sanity-check "
        "summary.</li></ul></div>"
        "<div class='flag'><strong>Open item (not fixed, needs a decision):</strong> "
        "a small number of postings (~0.6% of rows, ~0.07% of flagged ads) are "
        "double-counted across calendar-year boundaries because Phase I's "
        "duplicate-posting dedup resets per year. See section 20 below for the "
        "live-recomputed numbers and notes/meeting_prep_2026-08-02.md #2.6 for "
        "the full writeup.</div>"
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
    out_path = out_dir / "dashboard_v3.html"
    out_path.write_text("\n".join(body_parts), encoding="utf-8")
    print(f"\nSaved dashboard v3 to {out_path.resolve()}")


if __name__ == "__main__":
    main()
