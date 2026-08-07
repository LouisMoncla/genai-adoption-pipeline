"""
V4 results dashboard - first look at the just-completed 117-keyword full
reclassification (49 originally validated + 68 merged in from the raw
layer1/2/3 candidate pools, see code/mine/expand_master_keywords.py).
Deliberately does NOT touch v7_scoring.py's dynamic G1 (zero-pre-period)/
c-threshold/benchmark methodology - this dashboard only uses
master_keywords.json's static per-keyword "group" field (1/2/3) and
pipeline.group_classification's matching functions, same boundary
build_results_dashboard_v3.py already keeps between its Section 18/19
(group-based) and anything v7-shaped.

Three sections, in the order asked for:
  1. Raw per-keyword match volume as bar charts ("who matches what") - full
     pool, and the 68 new keywords on their own so they're easy to scan
     against the original 49's baseline.
  2. Elbow plot: G1 is treated as a strong flag on its own (no threshold),
     Group 2 + Group 3 combined swept over a corroboration threshold k -
     exactly v3's existing "Option B" (b_flags = (g1>=1) | (g23>=k)),
     rerun on the new, much larger flagged population. k=3 was "confirmed
     by Jeremias" for the OLD 49-keyword system in v3 - not assumed still
     true here, that's part of what this rerun is meant to re-examine.
  3. False-positive risk flagging - qualitative, informed by this
     dashboard's own computed volumes plus what's already been established
     this session about specific risky terms. Our opinion for discussion,
     not a keep/cut decision - that's the second (not-yet-built) dashboard.

KNOWN WRINKLE (see build_results_dashboard_v3.py's _recompute_full_matches
docstring): the persisted matched_group_keywords column only keeps the
WINNING (highest-priority) group's keyword(s) per ad
(group_classification.py's _classify_file_worker: best_group = min(...),
triggers filtered to that group only) - an ad matching both a Group 2 and
Group 3 keyword loses the Group 3 one in that column. Fine for the raw
per-keyword volume counts in Section 1 (matches v3's own existing
behavior/limitation there), NOT fine for Section 2's G2+G3 elbow - which is
exactly why _recompute_full_matches() re-scans ad text directly instead of
trusting that column, same as v3 already does.
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

B_THRESHOLDS = [1, 2, 3, 4, 5, 6]
LAYER_LABELS = {"St": "St (layer1_stanford.py)", "H&L": "H&L (layer2_hosseini.py)", "LLM": "LLM (layer3.py)"}


# --- Overview charts, reused from build_results_dashboard_v3.py (unchanged
# logic - these operate generically on the group/tst_created/detected_language
# columns, so they work as-is against the new 117-keyword classified_data;
# only title text updated for the actual 2016-2025 range) ---

def chart_classification_breakdown(df: pl.DataFrame):
    counts = df.select("group").group_by("group").len().sort("group").to_pandas()
    counts["label"] = counts["group"].map({
        "1": "Group 1 (core GenAI)", "2": "Group 2", "3": "Group 3", "NA": "No keyword match",
    })
    fig = px.pie(
        counts, names="label", values="len", hole=0.45,
        title="Overall classification breakdown (all postings, 2016-2025)",
        color="label",
        color_discrete_map={
            "Group 1 (core GenAI)": "#2ca02c", "Group 2": "#1f77b4",
            "Group 3": "#ff7f0e", "No keyword match": "#d3d3d3",
        },
    )
    fig.update_traces(textinfo="label+percent+value")
    return fig


def chart_matches_per_group(df: pl.DataFrame):
    counts = (
        df.select("group").filter(pl.col("group") != "NA")
        .group_by("group").len().sort("group")
        .to_pandas()
    )
    counts["label"] = "Group " + counts["group"]
    fig = px.bar(
        counts, x="label", y="len", color="label", text="len",
        title="Matches per group (Group 1/2/3 counts, 117-keyword pool)",
        labels={"label": "", "len": "Number of postings"},
        color_discrete_map={"Group 1": "#2ca02c", "Group 2": "#1f77b4", "Group 3": "#ff7f0e"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


def chart_share_over_time(df: pl.DataFrame):
    # Select down to just the 2 needed columns, and group by cheap integer
    # year/month (not a full-corpus dt.strftime() string column - that
    # allocates a new string per row across all 6.3M rows while every other
    # heavy column, content_clean/matched_group_keywords, is still attached
    # to the frame; on this machine's tight RAM that segfaulted polars'
    # native allocator. The "YYYY-MM" string is built only on the tiny
    # post-groupby table instead.
    monthly = (
        df.select(["tst_created", "group"])
        .with_columns([
            pl.col("tst_created").dt.year().alias("year"),
            pl.col("tst_created").dt.month().alias("month_num"),
        ])
        .group_by(["year", "month_num", "group"]).len()
    )
    monthly = monthly.with_columns(
        (pl.col("year").cast(pl.Utf8) + "-" + pl.col("month_num").cast(pl.Utf8).str.zfill(2)).alias("month")
    ).sort(["year", "month_num"])
    pivot = monthly.pivot(on="group", index="month", values="len", maintain_order=True).fill_null(0)
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
        title="GenAI-mention share of job postings over time (2016-2025)",
        labels={"month": "", "share_pct": "% of postings", "series": ""},
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def _group_year_pivot(df: pl.DataFrame) -> pl.DataFrame:
    yearly = (
        df.select(["tst_created", "group"])
        .with_columns(pl.col("tst_created").dt.year().alias("year"))
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
        title="Group share per year (each group's % of that year's postings)",
        labels={"year": "", "share_pct": "% of that year's postings", "group": ""},
        color_discrete_map={"Group 1": "#2ca02c", "Group 2": "#1f77b4", "Group 3": "#ff7f0e"},
    )
    return fig


def chart_language_comparison(df: pl.DataFrame):
    slim = df.select(["detected_language", "group"])
    total = slim.height
    corpus = (
        slim["detected_language"].value_counts()
        .with_columns((100 * pl.col("count") / total).alias("share"))
        .with_columns(pl.lit("All postings").alias("scope"))
    )
    flagged_df = slim.filter(pl.col("group") != "NA")
    ftotal = flagged_df.height
    flagged = (
        flagged_df["detected_language"].value_counts()
        .with_columns((100 * pl.col("count") / ftotal).alias("share"))
        .with_columns(pl.lit("Flagged postings").alias("scope"))
    )
    combined = pl.concat([corpus, flagged]).to_pandas()
    fig = px.bar(
        combined, x="detected_language", y="share", color="scope", barmode="group",
        title="Language distribution: all postings vs. GenAI-flagged postings (117-keyword pool)",
        labels={"detected_language": "", "share": "% share", "scope": ""},
    )
    return fig


def _recompute_full_matches(df: pl.DataFrame, en_patterns, local_patterns) -> pl.DataFrame:
    """Same logic as build_results_dashboard_v3.py's function of the same
    name - the persisted `group` column only keeps the single best-priority
    group's keywords, so the elbow plot needs the full per-group recount on
    just the already-flagged ads (cheap: ~71K ads, not the full 6.3M)."""
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


def keyword_volume_counts(df: pl.DataFrame, master_df: pl.DataFrame) -> pl.DataFrame:
    """Per-keyword ad-match count from the persisted matched_group_keywords
    column - same source/limitation as v3's table_keyword_coverage()."""
    counts = (
        df.select("matched_group_keywords")
        .explode("matched_group_keywords")
        .drop_nulls()
        .rename({"matched_group_keywords": "keyword"})
        .group_by("keyword")
        .agg(pl.len().alias("ad_match_count"))
    )
    return (
        master_df.join(counts, on="keyword", how="left")
        .with_columns(pl.col("ad_match_count").fill_null(0))
        .sort("ad_match_count", descending=True)
    )


def chart_keyword_volume(table: pl.DataFrame, title: str):
    df = table.filter(pl.col("ad_match_count") > 0).to_pandas()
    fig = px.bar(
        df.sort_values("ad_match_count"), x="ad_match_count", y="keyword", color="group",
        orientation="h", title=title,
        labels={"ad_match_count": "# ads matched", "keyword": "", "group": "Group"},
        color_discrete_map={1: "#2ca02c", 2: "#1f77b4", 3: "#c9c9c9"},
        category_orders={"group": [1, 2, 3]},
    )
    fig.update_layout(height=max(500, 22 * len(df)), legend_title_text="Group")
    return fig


def table_html(table: pl.DataFrame) -> str:
    rows = "".join(
        f"<tr><td style='padding:4px;border-top:1px solid #eee'>{r['keyword']}</td>"
        f"<td style='padding:4px;border-top:1px solid #eee'>{r['group']}</td>"
        f"<td style='padding:4px;border-top:1px solid #eee'>{LAYER_LABELS.get(r['source'], r['source'])}</td>"
        f"<td style='padding:4px;border-top:1px solid #eee'>{'validated' if r['validated'] else 'new (unvalidated)'}</td>"
        f"<td style='padding:4px;border-top:1px solid #eee;text-align:right'>{r['ad_match_count']:,}</td></tr>"
        for r in table.iter_rows(named=True)
    )
    return (
        "<details><summary style='cursor:pointer'>Full 117-keyword table (click to expand)</summary>"
        "<table style='border-collapse:collapse;width:100%;margin-top:10px;font-size:13px'>"
        "<tr style='background:#f0f0f0'><th style='text-align:left;padding:4px'>Keyword</th>"
        "<th style='text-align:left;padding:4px'>Group</th><th style='text-align:left;padding:4px'>Layer</th>"
        "<th style='text-align:left;padding:4px'>Status</th>"
        "<th style='text-align:right;padding:4px'>Ad matches</th></tr>"
        + rows + "</table></details>"
    )


def build_elbow_section(df: pl.DataFrame, master_keywords: list) -> tuple:
    flagged = df.filter(pl.col("group") != "NA").select(["ad_id", "content_clean", "detected_language"])
    en_patterns, local_patterns = build_keyword_patterns(master_keywords)
    matches = _recompute_full_matches(flagged, en_patterns, local_patterns)

    g1, g2, g3 = matches["g1_count"], matches["g2_count"], matches["g3_count"]
    g23 = g2 + g3
    high_conf = g1 >= 1

    b_flags = {k: ((g1 >= 1) | (g23 >= k)) for k in B_THRESHOLDS}
    elbow_pdf = pd.DataFrame({
        "k": B_THRESHOLDS,
        "total_flagged": [int(b_flags[k].sum()) for k in B_THRESHOLDS],
        "newly_swept_in": [int((b_flags[k] & ~high_conf).sum()) for k in B_THRESHOLDS],
    })
    elbow_fig = px.line(
        elbow_pdf.melt(id_vars="k", value_vars=["total_flagged", "newly_swept_in"],
                        var_name="series", value_name="ads"),
        x="k", y="ads", color="series", markers=True,
        title="Option B elbow plot (117-keyword pool): G1 flags alone; G2+G3 combined swept over k",
        labels={"k": "k (min combined Group 2+3 keywords, when no Group 1 match)", "ads": "# ads flagged"},
    )
    elbow_fig.update_xaxes(dtick=1)

    driver_html = ""
    for k in (2, 3):
        newly = b_flags[k] & ~high_conf
        freq = _keyword_freq(matches, newly)
        denom = int(newly.sum())
        if not freq:
            driver_html += f"<h3>k={k}</h3><p><i>No newly-swept-in ads at this k.</i></p>"
            continue
        top_kw, top_count = freq.most_common(1)[0]
        share = top_count / denom if denom else 0
        rows = "".join(
            f"<tr><td style='padding:4px'>{kw}</td><td style='padding:4px'>{n:,}</td>"
            f"<td style='padding:4px'>{n/denom*100:.0f}%</td></tr>"
            for kw, n in freq.most_common(10)
        )
        warn = (
            f"<p style='color:#a15c00'><strong>FLAG:</strong> '{top_kw}' alone drives "
            f"{share:.0%} of the {denom:,} newly-swept-in ads at k={k} - not really "
            f"independent multi-signal corroboration.</p>" if share > 0.5 else ""
        )
        driver_html += (
            f"<h3>k={k} &mdash; {denom:,} newly swept in beyond G1-alone</h3>"
            "<table style='border-collapse:collapse'><tr style='background:#f0f0f0'>"
            "<th style='padding:4px;text-align:left'>Keyword</th><th style='padding:4px;text-align:left'>Ads driven</th>"
            "<th style='padding:4px;text-align:left'>% of newly-swept-in</th></tr>"
            + rows + "</table>" + warn
        )

    summary_html = (
        f"<p>Population: {matches.height:,} ads with &ge;1 of the 117 keywords matched "
        f"(<code>group != \"NA\"</code>). High-confidence (&ge;1 Group 1 keyword): "
        f"{int(high_conf.sum()):,}.</p>"
        + elbow_pdf.to_html(index=False, border=0)
        + driver_html
    )
    return elbow_fig, summary_html


def build_false_positive_section(table: pl.DataFrame) -> str:
    total_matched = table.filter(pl.col("ad_match_count") > 0)["ad_match_count"].sum()
    new_kws = table.filter(~pl.col("validated")).sort("ad_match_count", descending=True)

    risky_generic = {
        "Artificial Intelligence", "Machine Learning", "Neural Networks", "Chatbot",
        "Deep Learning", "Digital Transformation", "Natural Language Processing",
        "Virtual Assistant", "Intelligent Automation", "AI Strategy", "AI-Powered",
    }
    risky_lexical = {
        "Transformer": "collides with electrical transformers in unrelated (e.g. industrial/electrical) job ads",
        "Falcon": "collides with aircraft models and unrelated brand names",
        "Jasper": "a common personal name",
        "Fine-Tuning": "generic engineering/audio/mechanical term outside ML context (partially mitigated - EN loanword confirmed used as-is in DE/FR/IT ads too, per Louis's check)",
        "Prompt Engineer": "job-title phrasing risk - lower concern after the DE loanword fix",
    }

    rows = []
    for r in new_kws.iter_rows(named=True):
        kw, n, grp = r["keyword"], r["ad_match_count"], r["group"]
        share = n / total_matched * 100 if total_matched else 0
        reasons = []
        if kw in risky_generic:
            reasons.append("very generic term, likely matches ordinary tech job ads with no real GenAI signal")
        if kw in risky_lexical:
            reasons.append(risky_lexical[kw])
        if not reasons and n > 0 and share > 2:
            reasons.append(f"high volume ({share:.1f}% of all matches) for a Group {grp}, unvalidated term - worth a manual eyeball")
        if reasons:
            rows.append((kw, grp, n, share, "; ".join(reasons)))

    if not rows:
        body = "<p><i>Nothing stood out beyond the earlier-flagged terms.</i></p>"
    else:
        body = (
            "<table style='border-collapse:collapse;width:100%'>"
            "<tr style='background:#f0f0f0'><th style='padding:6px;text-align:left'>Keyword</th>"
            "<th style='padding:6px;text-align:left'>Group</th><th style='padding:6px;text-align:right'>Ad matches</th>"
            "<th style='padding:6px;text-align:right'>% of all matches</th>"
            "<th style='padding:6px;text-align:left'>Why it's worth a second look</th></tr>"
            + "".join(
                f"<tr><td style='padding:6px'>{kw}</td><td style='padding:6px'>{grp}</td>"
                f"<td style='padding:6px;text-align:right'>{n:,}</td>"
                f"<td style='padding:6px;text-align:right'>{share:.2f}%</td>"
                f"<td style='padding:6px'>{reason}</td></tr>"
                for kw, grp, n, share, reason in rows
            ) + "</table>"
        )

    return (
        "<p style='color:#666'>Our opinion, for discussion - not a keep/cut decision. The actual "
        "decision on which of the 68 new keywords to keep is the second dashboard, not this one.</p>"
        + body
    )


def main():
    parser = argparse.ArgumentParser(description="Build the v4 raw-results dashboard (117-keyword reclassification)")
    parser.add_argument("--config", type=str, default="pipeline/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    master_keywords = load_master_keywords()
    master_df = pl.DataFrame([
        {"keyword": k["keyword"], "group": k["group"], "source": k["source"], "validated": k["validated"]}
        for k in master_keywords
    ])

    print("Loading classified_data (117-keyword reclassification)...")
    df = pl.scan_parquet(Path(config.output_dir) / "classified_data" / "year=*" / "*.parquet").collect()
    total_ads = df.height
    n_flagged = df.filter(pl.col("group") != "NA").height
    print(f"  {total_ads:,} total ads, {n_flagged:,} flagged (group != NA), "
          f"{100*n_flagged/total_ads:.4f}%.")

    print("Section 0: overview charts (reused from build_results_dashboard_v3.py)...")
    print("  0a: matches_per_group...")
    c0a = chart_matches_per_group(df)
    print("  0b: classification_breakdown...")
    c0b = chart_classification_breakdown(df)
    print("  0c: group_share_per_year...")
    c0c = chart_group_share_per_year(df)
    print("  0d: share_over_time...")
    c0d = chart_share_over_time(df)
    print("  0e: language_comparison...")
    c0e = chart_language_comparison(df)
    print("  0f: done.")
    overview_charts = [c0a, c0b, c0c, c0d, c0e]

    print("Section 1: per-keyword match volume...")
    vol_table = keyword_volume_counts(df, master_df)
    full_chart = chart_keyword_volume(vol_table, "All 117 keywords: ad match volume by group")
    new_chart = chart_keyword_volume(
        vol_table.filter(~pl.col("validated")), "The 68 new (unvalidated) keywords: ad match volume by group"
    )

    print("Section 2: Option B elbow plot (G1 strong flag, G2+G3 vs k)...")
    elbow_fig, elbow_html = build_elbow_section(df, master_keywords)

    print("Section 3: false-positive risk flagging...")
    fp_html = build_false_positive_section(vol_table)

    print("Assembling dashboard...")
    sections = []
    overview_titles = [
        "Overview &mdash; matches per group",
        "Overview &mdash; classification breakdown",
        "Overview &mdash; group share per year",
        "Overview &mdash; GenAI-mention share over time",
        "Overview &mdash; language distribution (all vs. flagged)",
    ]
    for title, fig in zip(overview_titles, overview_charts):
        sections.append((title, "figure", fig))
    sections.append((
        "Raw per-keyword match volume &mdash; full pool", "figure", full_chart,
    ))
    sections.append((
        "Raw per-keyword match volume &mdash; new 68 keywords only", "figure", new_chart,
    ))
    sections.append((
        "Full keyword table", "html", table_html(vol_table),
    ))
    sections.append((
        "Elbow plot: G1 strong flag, G2+G3 combined vs. k", "figure", elbow_fig,
    ))
    sections.append((
        "Elbow plot detail &amp; newly-swept-in drivers", "html", elbow_html,
    ))
    sections.append((
        "False-positive risk flagging (our opinion, for discussion)", "html", fp_html,
    ))

    def _nav_label(title: str) -> str:
        if title.startswith("Overview &mdash;"):
            return title.replace("Overview &mdash; ", "Overview: ")
        return title.split(" &mdash;")[0].split(" (")[0]

    nav_links = "".join(
        f'<a href="#s{i}">{i}. {_nav_label(title)}</a>'
        for i, (title, _kind, _payload) in enumerate(sections, start=1)
    )

    body_parts = [
        """<html><head><meta charset='utf-8'>
        <title>v4 Results Dashboard - 117-Keyword Reclassification</title>
        <style>
        :root{--ink:#1a1a2e;--muted:#666;--accent:#3b5bdb;--bg-soft:#f7f8fb;--border:#e1e4ea;}
        body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
             max-width:1050px;margin:0 auto;padding:24px 28px 80px;color:var(--ink);line-height:1.55;}
        h1{margin-bottom:2px;font-size:26px;}
        h2{font-size:19px;margin-top:0;border-bottom:2px solid var(--accent);padding-bottom:6px;}
        h3{font-size:15px;color:var(--ink);margin-top:22px;margin-bottom:8px;}
        .subtitle{color:var(--muted);margin-top:4px;margin-bottom:28px;font-size:14px;}
        .chart{margin-bottom:20px;padding:22px 26px;background:var(--bg-soft);
               border:1px solid var(--border);border-radius:10px;}
        table{font-size:13px;}
        nav.toc{position:sticky;top:0;background:#fff;padding:10px 0;margin-bottom:10px;
                border-bottom:1px solid var(--border);font-size:13px;z-index:10;}
        nav.toc a{margin-right:16px;color:var(--accent);text-decoration:none;}
        nav.toc a:hover{text-decoration:underline;}
        </style></head><body>""",
        "<h1>v4 Results Dashboard &mdash; 117-Keyword Reclassification</h1>",
        f"<p class='subtitle'>Raw results only - does not use <code>v7_scoring.py</code>'s dynamic "
        f"G1/c-threshold methodology, just <code>master_keywords.json</code>'s static Group 1/2/3 "
        f"tiers. {total_ads:,} total ads, {n_flagged:,} flagged ({100*n_flagged/total_ads:.4f}%, "
        f"vs. 0.1152% / 7,263 ads with the original 49 keywords).</p>",
        f"<nav class='toc'>{nav_links}</nav>",
    ]

    first = True
    for i, (title, kind, payload) in enumerate(sections, start=1):
        if kind == "html":
            body_parts.append(f"<div class='chart' id='s{i}'><h2>{i}. {title}</h2>{payload}</div>")
            continue
        html = pio.to_html(payload, full_html=False, include_plotlyjs="inline" if first else False)
        first = False
        body_parts.append(f"<div class='chart' id='s{i}'><h2>{i}. {title}</h2>{html}</div>")

    body_parts.append("</body></html>")

    out_dir = Path(config.output_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_dashboard_v4.html"
    out_path.write_text("\n".join(body_parts), encoding="utf-8")
    print(f"\nSaved v4 results dashboard to {out_path.resolve()}")


if __name__ == "__main__":
    main()
