"""
v7 scoring results dashboard - single-file HTML, same self-contained
convention as build_results_dashboard_v3.py (inline Plotly, no external
deps, double-click to open). Everything is computed fresh from
output/classified_data each run (not read back from the CSVs
v7_scoring.py separately saves) - avoids any risk of the dashboard drifting
out of sync with a stale CSV; v7_scoring.py's own runs are cheap enough
(under a minute each) that recomputing three times here is not a real cost.

Covers: the full-keyword-pool grid sweep and the Layer-3-excluded
sensitivity run (both the k-axis and c-axis elbow views), the k=2-vs-k=3
comparison, and the full per-keyword classification table. See
notes/phase_b_v7_scoring_2026-08-03.md for the earlier narrative this
dashboard visualizes (some of it now superseded - see the 2026-08-06
changes below).

MATH RENDERING (added 2026-08-04, per request for "more professional,
LaTeX-style" explanations): formulas are rendered via matplotlib's mathtext
(a LaTeX-like parser bundled with matplotlib - no real LaTeX install or
internet connection needed) to small transparent PNGs, base64-embedded
inline as <img> tags. Same self-contained philosophy as embedding Plotly
inline rather than loading it from a CDN - this dashboard should open and
look right on a machine with no internet access.
"""

import argparse
import base64
import io
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio
import polars as pl

from pipeline.config_loader import load_config
from pipeline.group_classification import build_keyword_patterns, load_master_keywords

from v7_scoring import (
    C_GRID, G1_VALIDATION_YEARS, K_GRID, LANGS, VALIDATION_YEAR,
    collect_matches, classify_keywords, g2_keywords_at_c, run_grid_sweep,
)

EXAMPLE_C = 5  # per Jeremias, 2026-08-06 - re-picked after C_GRID moved to
               # [1.5, 3, 5, 10] (old EXAMPLE_C=1.20 isn't even in the new
               # grid); 5 sits mid-range. Just an illustrative point for the
               # G2-candidate table/prose, not itself a decision.
CANDIDATE_KS = [2, 3]  # the two values under active discussion as of
                       # 2026-08-04 - shown side by side (c-sensitivity
                       # charts, newly-swept-in driver-keyword breakdown) so
                       # Jeremias has the full picture to pick k himself,
                       # not just our own leaning.

# Per Jeremias, 2026-08-06: drop these two from this rerun entirely - bare
# "Transformer" (electrical-transformer collision risk, flagged since the
# layer1/2/3 merge) and "Digital Transformation" (the single most generic
# term in the pool - 31% of ALL keyword matches in the v4 dashboard, and its
# breakeven_c=4.49 was already the highest of any newly-merged keyword,
# meaning nothing in the old c grid could exclude it anyway).
MANUALLY_EXCLUDED = {"Digital Transformation", "Transformer"}


def latex(formula: str, fontsize: int = 15, color: str = "#1a1a1a", display: bool = False) -> str:
    """Render a LaTeX-style formula (matplotlib mathtext) to a base64 PNG,
    returned as a ready-to-embed <img> tag. `display=True` renders slightly
    larger and wraps in a centered block, for standalone equations rather
    than inline symbols within a sentence."""
    size = fontsize + 4 if display else fontsize
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.patch.set_alpha(0)
    fig.text(0, 0, f"${formula}$", fontsize=size, color=color)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    style = "vertical-align:middle;max-width:100%;" if not display else "vertical-align:middle;"
    img = f'<img src="data:image/png;base64,{b64}" style="{style}" alt="{formula}">'
    if display:
        return f'<div style="text-align:center;margin:14px 0">{img}</div>'
    return img


def _kw_summary_df(kw_stats: dict, g2_kws: set) -> pd.DataFrame:
    """One row per keyword, only the numbers that actually matter for its
    OWN classification - replaces the old 12-column (B/y2025/multiple x 4
    languages) table, most of which was zeros for any given keyword since
    real signal usually lives in only 1-2 languages. For G2 candidates, only
    the "best" language (the one with the highest multiple - the one that
    actually determines G2 status, since G2 only needs ONE language to
    qualify) is shown."""
    rows = []
    for kw, s in kw_stats.items():
        if s["is_g1"]:
            status = "G1"
        elif kw in g2_kws:
            status = "G2"
        else:
            status = "neither"

        best_lang, best_multiple, best_y2025, best_b, best_mean = None, None, None, None, None
        for lang, stats in s["per_lang"].items():
            multiple = stats["y2025"] / stats["B"] if stats["B"] > 0 else (
                float("inf") if stats["y2025"] > 0 else 0.0
            )
            if best_multiple is None or multiple > best_multiple:
                best_lang, best_multiple, best_y2025, best_b, best_mean = (
                    lang, multiple, stats["y2025"], stats["B"], stats["mean"]
                )

        rows.append({
            "keyword": kw,
            "is_g1": s["is_g1"],
            "status": status,
            "mean_pre2020": round(best_mean, 2) if best_mean is not None else None,
            "y2025_best_lang": best_y2025,
            "y2025_all_langs": s["y2025_all_langs"],
            "best_language": best_lang,
            "B_best_lang": round(best_b, 1) if best_b is not None else None,
            "breakeven_c": (
                round(best_multiple, 2) if status != "G1" and best_multiple not in (None, float("inf")) else
                ("inf" if best_multiple == float("inf") and status != "G1" else
                 "G1 (n/a - flags on its own)" if status == "G1" else None)
            ),
        })
    df = pd.DataFrame(rows)
    status_order = {"G1": 0, "G2": 1, "neither": 2}
    df["_sort"] = df["status"].map(status_order)
    return df.sort_values(["_sort", "y2025_all_langs"], ascending=[True, False]).drop(columns="_sort")


def chart_keyword_ranking(summary_df: pd.DataFrame, title: str, top_n: int = 49):
    """Horizontal bar of every keyword's 2025 volume, colored by G1 / G2 /
    neither - makes "what's actually driving the numbers" scannable instead
    of requiring a row-by-row table read."""
    df = summary_df.sort_values("y2025_all_langs", ascending=False).head(top_n)
    fig = px.bar(
        df.sort_values("y2025_all_langs"), x="y2025_all_langs", y="keyword", color="status",
        orientation="h", title=title,
        labels={"y2025_all_langs": "2025 mentions (all languages)", "keyword": "", "status": ""},
        color_discrete_map={"G1": "#2ca02c", "G2": "#1f77b4", "neither": "#c9c9c9"},
        category_orders={"status": ["G1", "G2", "neither"]},
    )
    # ~24px/bar so every keyword's y-axis label has room to render - a fixed
    # height (previously 850, ~17px/bar for 49 keywords) was too cramped and
    # Plotly silently dropped some tick labels to avoid overlap.
    fig.update_layout(height=max(600, 24 * len(df)), legend_title_text="")
    return fig


def table_html(df: pd.DataFrame, max_rows: int = 60) -> str:
    return df.head(max_rows).to_html(index=False, border=0, classes="datatable")


_FULL_TABLE_COLS = [
    "keyword", "matched_text", "is_g1", "status", "mean_pre2020", "y2025_best_lang",
    "y2025_all_langs", "y2023_2025", "B_best_lang", "breakeven_c", "flagged_k2", "flagged_k3",
]
_NUM_FMT = {
    "mean_pre2020": lambda v: f"{v:,.1f}" if v is not None else "-",
    "y2025_best_lang": lambda v: f"{v:,}" if v is not None else "-",
    "y2025_all_langs": lambda v: f"{v:,}",
    "y2023_2025": lambda v: f"{v:,}",
    "B_best_lang": lambda v: f"{v:,.1f}" if v is not None else "-",
    "flagged_k2": lambda v: f"{v:,}",
    "flagged_k3": lambda v: f"{v:,}",
}


def _fmt_row_cells(row: dict) -> str:
    cells = []
    for col in _FULL_TABLE_COLS:
        val = row[col]
        fmt = _NUM_FMT.get(col)
        cells.append(str(fmt(val)) if fmt else str(val))
    return "".join(f"<td>{c}</td>" for c in cells)


def full_keyword_table_html(full_table: pd.DataFrame, max_c: float) -> str:
    """Per Louis, 2026-08-06: zero-2025-match keywords were sorting to the
    very top of the breakeven_c-ascending order (multiple = 0/B = 0, the
    lowest possible value), crowding out the genuinely interesting
    low-but-nonzero borderline candidates. Split into three explicit,
    labeled groups instead of one flat sort: (1) has real 2025 signal, sorted
    by breakeven_c ascending - the actually useful "which to cut" ordering;
    (2) G1 keywords, immune to c; (3) zero-2025-match keywords, alphabetical
    - present in the master list but contributing nothing this year, exiled
    to the bottom so they stop diluting group (1).
    Also: sticky header (117 rows scrolls past the column names otherwise),
    thousands-separator formatting, and a highlight on rows whose
    breakeven_c actually falls inside the tested c grid (<= max_c) - those
    are the ones that would really flip status somewhere in this dashboard's
    own sweep, not just in theory.
    matched_text (2026-08-06, per the LLM/G1 confusion): a keyword's
    display NAME isn't always what actually gets matched against ad text -
    "LLM" and "RAG" are both deliberately repointed to a spelled-out phrase
    (build_master_keywords.py's FORM_OVERRIDES, avoiding LL.M./RAG-the-
    auditor-designation collisions). Blank whenever name == matched text
    (115 of 117 rows), so it only draws the eye where it actually matters."""
    has_signal = full_table[(~full_table["is_g1_bool"]) & (full_table["y2025_all_langs"] > 0)]
    has_signal = has_signal.sort_values(["breakeven_c_num", "y2025_all_langs"], ascending=[True, False])
    g1_rows = full_table[full_table["is_g1_bool"]].sort_values("y2025_all_langs", ascending=False)
    no_match = full_table[(~full_table["is_g1_bool"]) & (full_table["y2025_all_langs"] == 0)]
    no_match = no_match.sort_values("keyword")

    header = "<tr>" + "".join(f"<th>{c}</th>" for c in _FULL_TABLE_COLS) + "</tr>"

    def _group_rows(df, highlight=False):
        out = []
        for _, r in df.iterrows():
            row = r.to_dict()
            in_tested_range = highlight and isinstance(row["breakeven_c"], (int, float)) and row["breakeven_c"] <= max_c
            cls = " class='would-cut'" if in_tested_range else ""
            out.append(f"<tr{cls}>{_fmt_row_cells(row)}</tr>")
        return "".join(out)

    body = (
        f"<tr class='group-break'><td colspan='{len(_FULL_TABLE_COLS)}'>"
        f"Has 2025 signal ({len(has_signal)}) &mdash; sorted by breakeven_c, weakest first. "
        f"Rows shaded amber would actually flip to &ldquo;excluded&rdquo; somewhere within the "
        f"tested c grid (&le;{max_c:g}).</td></tr>"
        + _group_rows(has_signal, highlight=True)
        + f"<tr class='group-break'><td colspan='{len(_FULL_TABLE_COLS)}'>"
        f"G1 keywords ({len(g1_rows)}) &mdash; immune to c, flag on their own.</td></tr>"
        + _group_rows(g1_rows)
        + f"<tr class='group-break'><td colspan='{len(_FULL_TABLE_COLS)}'>"
        f"No 2025 matches at all ({len(no_match)}) &mdash; in the master list, contributed "
        f"nothing this year.</td></tr>"
        + _group_rows(no_match)
    )

    return (
        "<div class='scroll-table'><table class='datatable'>"
        f"<thead>{header}</thead><tbody>{body}</tbody></table></div>"
    )


def chart_elbow(sweep_df: pd.DataFrame, title: str):
    """The actual elbow plot: x=k (the axis that matters - how many
    corroborating G2 keywords to require), y=% flagged, one line per c.
    This is the orientation code/mine/group_strategy_analysis.py's own
    Option B elbow plot already uses (x=k, y=flagged count) - matching it
    here rather than the previous c-on-x-axis layout, which put the
    low-variation axis in the position that's supposed to show the bend."""
    df = sweep_df.copy()
    df["c_label"] = df["c"].map(lambda x: f"c = {x:g}")
    fig = px.line(
        df, x="k", y="flagged_share_pct", color="c_label", markers=True,
        title=title,
        labels={"k": "k  (min_keywords_g2 - corroborating G2 keywords required)",
                "flagged_share_pct": "% of 2025 ads flagged", "c_label": ""},
    )
    fig.update_xaxes(dtick=1)
    fig.update_traces(line=dict(width=2), marker=dict(size=8))
    fig.update_layout(legend_title_text="")
    return fig


def chart_c_elbow(sweep_df: pd.DataFrame, title: str):
    """c on the x-axis, one line per k - the mirror image of chart_elbow().
    Dropped early in v7 (49-keyword pool, G2 candidate set too small for c
    to show much variation - the bend lived on the k-axis instead).
    Re-added now that the keyword pool is 117-strong: with far more G2
    candidates spread across a wider range of multiples, c's own effect is
    worth seeing directly rather than only through 5 separate k-fixed
    snapshots."""
    df = sweep_df.copy()
    df["k_label"] = df["k"].map(lambda x: f"k = {x}")
    fig = px.line(
        df, x="c", y="flagged_share_pct", color="k_label", markers=True,
        title=title,
        labels={"c": "c  (multiple_g2_threshold)", "flagged_share_pct": "% of 2025 ads flagged",
                "k_label": ""},
    )
    fig.update_xaxes(tickvals=sorted(df["c"].unique()))
    fig.update_traces(line=dict(width=2), marker=dict(size=8))
    fig.update_layout(legend_title_text="")
    return fig


def chart_c_sensitivity(sweep_df: pd.DataFrame, k: int, title: str):
    """Fixes k and shows what c alone does - the elbow plots put 5 c-lines
    on one chart (necessary to show the k-elbow), but that makes it hard to
    isolate c's own effect once a k has been picked. Only 5 points, so data
    labels are cheap and worth showing directly on the chart."""
    df = sweep_df[sweep_df["k"] == k].sort_values("c")
    fig = px.line(
        df, x="c", y="flagged_share_pct", markers=True, text="flagged_share_pct",
        title=title,
        labels={"c": "c  (multiple_g2_threshold)", "flagged_share_pct": "% of 2025 ads flagged"},
    )
    fig.update_traces(
        line=dict(width=2), marker=dict(size=9),
        texttemplate="%{text:.3f}%", textposition="top center",
    )
    fig.update_xaxes(tickvals=sorted(df["c"].unique()))
    return fig


def pivot_table_html(sweep_df: pd.DataFrame) -> str:
    """The (c, k) grid as a plain c x k matrix - rows=c, columns=k, cell =
    "count (pct%)" - instead of the original flat 25-row table (c outer / k
    inner, repeating), which was hard to scan, and instead of a color-coded
    heatmap (removed - not everyone reads color intensity well, and one
    shared scale made k=1's much-larger value wash out the k=2-5
    differences). Both the raw flagged count and the percentage in one
    cell, since the count is what was actually asked for and the percentage
    is still useful context for comparing across the different total-ad
    denominators elsewhere in this dashboard."""
    combined = sweep_df.copy()
    combined["cell"] = combined.apply(
        lambda r: f"{int(r['flagged_2025']):,} ({r['flagged_share_pct']:.3f}%)", axis=1
    )
    pct_pivot = combined.pivot(index="c", columns="k", values="cell").sort_index()
    pct_pivot.columns = [f"k={k}" for k in pct_pivot.columns]
    pct_html = pct_pivot.reset_index().to_html(index=False, border=0, classes="datatable", escape=False)

    g2_pivot = sweep_df.drop_duplicates("c")[["c", "n_g2_keywords"]].sort_values("c")
    g2_html = g2_pivot.rename(columns={"n_g2_keywords": "# G2 keywords"}).to_html(
        index=False, border=0, classes="datatable"
    )
    return (
        "<h3>Ads flagged: count (% of 2025 ads)</h3>" + pct_html +
        "<h3># keywords qualifying as G2 (doesn't vary by k)</h3>" + g2_html
    )


def chart_g2_count_vs_c(sweep_df: pd.DataFrame, title: str):
    """Pool-size sensitivity: how many keywords qualify as G2 at each c.
    Directly supports reading the elbow plots above - fewer G2 keywords
    at higher c naturally means fewer ads can reach a given k."""
    df = sweep_df.drop_duplicates("c")[["c", "n_g2_keywords"]].sort_values("c")
    fig = px.bar(
        df, x="c", y="n_g2_keywords", text="n_g2_keywords", title=title,
        labels={"c": "c  (multiple_g2_threshold)", "n_g2_keywords": "# keywords qualifying as G2"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(bargap=0.4)
    return fig


def elbow_flatten_note(sweep_df: pd.DataFrame) -> str:
    """For each c, find the first k (from k=3 onward) whose step-change from
    the previous k has dropped to <=10% of the initial k=1->k=2 drop - a
    simple, data-driven "where does this flatten" heuristic, not just an
    eyeballed claim."""
    rows = []
    for c in sorted(sweep_df["c"].unique()):
        sub = sweep_df[sweep_df["c"] == c].sort_values("k")
        vals = sub["flagged_share_pct"].tolist()
        ks = sub["k"].tolist()
        if len(vals) < 3:
            continue
        first_delta = abs(vals[1] - vals[0])
        elbow_k = ks[-1]
        for i in range(2, len(vals)):
            delta = abs(vals[i] - vals[i - 1])
            if first_delta > 0 and delta <= 0.1 * first_delta:
                elbow_k = ks[i]
                break
        rows.append((c, elbow_k))
    by_k = {}
    for c, k in rows:
        by_k.setdefault(k, []).append(c)
    parts = [
        f"k={k} (c={', '.join(f'{c:g}' for c in cs)})" if len(cs) < len(rows) else f"k={k} (all c)"
        for k, cs in sorted(by_k.items())
    ]
    return (
        f"<p style='color:#666;font-size:13px'><b>Where it flattens</b> (first k where the "
        f"step-change drops to &le;10% of the k=1&rarr;k=2 drop): {'; '.join(parts)}.</p>"
    )


def flagged_set_at(stats: dict, matches: list, k: int, c: float = EXAMPLE_C) -> set:
    g1 = {kw for kw, s in stats.items() if s["is_g1"]}
    g2 = g2_keywords_at_c(stats, c)
    out = set()
    for ad_id, year, kws in matches:
        if year != VALIDATION_YEAR:
            continue
        n_g1 = sum(1 for kw in kws if kw in g1)
        n_g2 = sum(1 for kw in kws if kw in g2)
        if n_g1 >= 1 or n_g2 >= k:
            out.add(ad_id)
    return out


def per_keyword_flagged_counts(stats: dict, matches: list, k: int, c: float = EXAMPLE_C) -> dict:
    """For each keyword: how many 2025 ads it matched that ALSO ended up in
    the final ad-level flagged set at this (c, k) - not the same as its raw
    y2025 count, since a G2 keyword's ads only count if the ad had enough
    other corroborating keywords too."""
    flagged = flagged_set_at(stats, matches, k, c)
    counts: dict = {}
    for ad_id, year, kws in matches:
        if year != VALIDATION_YEAR or ad_id not in flagged:
            continue
        for kw in kws:
            counts[kw] = counts.get(kw, 0) + 1
    return counts


def per_keyword_recent_volume(counts: dict, all_kws: list, years=G1_VALIDATION_YEARS) -> dict:
    """Sum of ad-match counts across the given years, all languages - a
    sense of sustained recent volume beyond the single 2025 validation
    year. Defaults to the same window as G1's own validation check
    (currently identical to a literal (2023,2024,2025) - reusing the
    constant instead of a second hardcoded literal so the two can't
    silently drift apart if G1_VALIDATION_YEARS ever changes)."""
    return {
        kw: sum(counts.get((y, l, kw), 0) for y in years for l in LANGS)
        for kw in all_kws
    }


def newly_swept_in_analysis(stats: dict, matches: list, k: int, c: float = EXAMPLE_C) -> dict:
    """For a given k: how many 2025 ads are flagged only because of G2
    corroboration (i.e. beyond what G1 alone already catches), and which
    keywords are actually driving those ads. The comparison baseline is
    "G1 alone" (N_G1 >= 1, ignoring G2 entirely), NOT k=1 - k=1 already
    includes any single G2 match on its own, so it's the loosest possible
    threshold and can never be exceeded by a larger k; comparing against it
    is meaningless (correcting a mistake made when first exploring this
    live - the fix matters enough to leave this note here)."""
    g1 = {kw for kw, s in stats.items() if s["is_g1"]}
    g2 = g2_keywords_at_c(stats, c)
    ads_2025 = [(ad_id, kws) for ad_id, year, kws in matches if year == VALIDATION_YEAR]
    kws_by_id = dict(ads_2025)

    g1_alone = {ad_id for ad_id, kws in ads_2025 if any(kw in g1 for kw in kws)}
    flagged_k = flagged_set_at(stats, matches, k, c)
    newly_swept_in = flagged_k - g1_alone

    driver_counts: dict = {}
    for ad_id in newly_swept_in:
        for kw in kws_by_id[ad_id]:
            if kw in g2:
                driver_counts[kw] = driver_counts.get(kw, 0) + 1
    drivers = sorted(driver_counts.items(), key=lambda x: -x[1])

    return {
        "k": k,
        "g1_alone_total": len(g1_alone),
        "flagged_total": len(flagged_k),
        "newly_swept_in": len(newly_swept_in),
        "drivers": drivers,
        "top_driver": drivers[0] if drivers else None,
    }


def build_methodology_section() -> str:
    """The full notation glossary + derivation, with matplotlib-rendered
    LaTeX. This is the section the "make it complete, explain every
    variable, use LaTeX" request is mainly about."""

    strategy_para = """
    <p class="lead">The core idea behind v7 is to test, for each candidate keyword, whether
    its use in 2025 job postings looks like a genuine structural shift rather than ordinary
    year-to-year noise. We build a <b>counterfactual benchmark</b> from the keyword's own
    history (2016&ndash;2020) &mdash; in effect asking &ldquo;how many mentions would we
    expect in 2025 if nothing unusual had happened?&rdquo; &mdash; and then check whether the
    actual 2025 count clears that benchmark by a wide enough margin, <i>and</i> reaches a
    large enough absolute number that it isn&rsquo;t just a handful of coincidental mentions.
    Keywords are split into two tiers based on how strong that evidence is: <b>G1</b> keywords
    literally did not appear in job postings before 2021 and now do &mdash; unambiguous enough
    that a single mention is trusted on its own. <b>G2</b> keywords existed before but have
    grown dramatically &mdash; individually weaker evidence, so an ad needs several distinct
    G2 keywords corroborating each other before it&rsquo;s trusted. An ad is finally flagged as
    GenAI-related if it clears either bar (or the special-case Copilot rule below).</p>
    """

    notation_rows = [
        ("t", "calendar year"),
        (r"\ell", "language: en, de, fr, or it"),
        (r"y_{t,\ell}", f"count of ads in year {latex('t')}, language {latex(r'\ell')}, containing this keyword"),
        (r"y_{2025,\ell}", f"shorthand for the 2025 count (the &ldquo;validation year&rdquo;) in language {latex(r'\ell')}"),
        (r"\hat{y}", "the OLS-fitted trend line, extrapolated to 2025"),
        (r"\bar{y}", "the flat average of the 2016&ndash;2020 counts"),
        ("B", "benchmark &mdash; the counterfactual &ldquo;expected&rdquo; 2025 count"),
        ("c", "<b>multiple_g2_threshold</b> &mdash; how many multiples of the benchmark 2025 must clear to count as G2 &nbsp;<span class='tag calibrate'>CALIBRATE</span>"),
        ("k", "<b>min_keywords_g2</b> &mdash; how many distinct G2 keywords an ad needs (when it has no G1 keyword) &nbsp;<span class='tag calibrate'>CALIBRATE</span>"),
        (r"N_{G1},\,N_{G2}", "count of distinct G1 / G2 keywords matched in a given ad"),
    ]
    notation_table = "".join(
        f"<tr><td class='sym'>{latex(sym)}</td><td>{meaning}</td></tr>"
        for sym, meaning in notation_rows
    )

    # Precomputed as variables (not typed inline as raw "$...$" text) so every
    # inline mention actually renders through the latex() rasterizer, same as
    # the big standalone display equations below - avoids the f-string brace-
    # escaping headache of calling latex() with \{...\} literals inline.
    lx_t = latex("t")
    lx_yt = latex("y_t")
    lx_trange = latex(r"t \in \{2016, \dots, 2020\}")
    lx_c = latex("c")
    lx_k = latex("k")
    lx_g2cond = latex(r"y_{2025,\ell} \geq c \cdot B_\ell")
    lx_bzero = latex(r"B_\ell = 0")

    steps = f"""
    <h3>1. Fitted trend</h3>
    <p>Ordinary least squares of the yearly counts {lx_yt} on the year {lx_t}, using only the five
    pre-period years {lx_trange} (2021 is deliberately excluded &mdash;
    GitHub Copilot&rsquo;s preview, DALL&middot;E, and the GPT-3 API all arrived that year, so
    2021 is treated as already-affected, not a clean baseline year). The fitted line is then
    extrapolated forward to 2025:</p>
    {latex(r"\hat{y} \;=\; \beta_0 \;+\; \beta_1 \cdot (2025 - 2016)", display=True)}
    <p style="color:#666">&ldquo;If the 2016&ndash;2020 trend had simply continued in a
    straight line, this is what 2025 would look like.&rdquo;</p>

    <h3>2. Baseline mean</h3>
    <p>A second, more conservative counterfactual &mdash; the flat historical average, ignoring
    any trend entirely:</p>
    {latex(r"\bar{y} \;=\; \frac{1}{5}\sum_{t=2016}^{2020} y_t", display=True)}

    <h3>3. Benchmark</h3>
    <p>Take whichever of the two counterfactuals is larger, and never let it go below zero
    (a keyword can&rsquo;t have had fewer than zero mentions, so a negative trend
    extrapolation shouldn&rsquo;t produce a negative benchmark):</p>
    {latex(r"B \;=\; \max(\hat{y},\, \bar{y},\, 0)", display=True)}

    <h3>4. G1 &mdash; &ldquo;didn&rsquo;t exist before, exists now&rdquo;</h3>
    <p>A fixed, binary property of the keyword as a whole &mdash; <i>not</i> evaluated
    per-language. Absent everywhere in every language throughout the whole pre-period, and
    mentioned at least once across {G1_VALIDATION_YEARS[0]}&ndash;{G1_VALIDATION_YEARS[-1]}
    combined, in any language <span style="color:#a15c00">(changed 2026-08-06, per Jeremias -
    was 2025 alone; using 3 years is more robust than hinging G1 status on a single year)</span>:</p>
    {latex(r"\left(\sum_{t=2016}^{2020}\sum_{\ell} y_{t,\ell} = 0\right) \;\wedge\; \left(\sum_{t=2023}^{2025}\sum_{\ell} y_{t,\ell} \geq 1\right)", display=True)}
    <p style="color:#666">No multiple is needed here &mdash; with an all-zero pre-period the
    benchmark is 0 by construction, so any multiple condition would hold automatically anyway.
    Only this &ge;1 check moved to the 3-year window; G2's own condition below (which still
    determines the benchmark/multiple math) is unchanged, still 2025 only.</p>

    <h3>5. G2 &mdash; &ldquo;existed before, grew sharply&rdquo;</h3>
    <p>Evaluated <i>per language</i>. True in just <b>one</b> language is enough to mark the
    keyword G2 everywhere (all of its language forms) &mdash; the mirror image of G1, which
    instead requires absence in <b>every</b> language:</p>
    {latex(r"y_{2025,\ell} \;\geq\; c \cdot B_{\ell} \quad\wedge\quad y_{2025,\ell} \geq 10", display=True)}
    <p style="color:#666">Two separate conditions, both required: 2025 usage beats its own
    benchmark by at least a factor of {lx_c} (the &ldquo;multiple&rdquo;), <i>and</i> reached at
    least 10 ads in that language &mdash; so a jump from 1 mention to 3 doesn&rsquo;t count,
    even though that&rsquo;s technically a 3&times; multiple. Implementation note: computed
    directly as {lx_g2cond} rather than dividing to get a
    &ldquo;multiple&rdquo; value first &mdash; avoids a division-by-zero when {lx_bzero},
    and correctly fails a keyword with literally no counts in any year.</p>

    <h3>6. Ad-level flag</h3>
    <p>An individual job ad is flagged as GenAI-related if <i>any</i> of three conditions
    holds:</p>
    {latex(r"N_{G1} \geq 1 \;\;\vee\;\; N_{G2} \geq k \;\;\vee\;\; \mathrm{RuleC}", display=True)}
    <p style="color:#666">One G1 keyword is sufficient on its own. Otherwise, {lx_k} distinct G2
    keywords must corroborate each other. <b>Rule C</b> is the special case: the word
    &ldquo;Copilot&rdquo; is matched <i>and</i> the ad text also contains &ldquo;Microsoft&rdquo;
    or &ldquo;GitHub&rdquo; &mdash; distinguishing Microsoft/GitHub Copilot from the aviation
    meaning of the word. Currently a structural no-op on this dataset &mdash; bare
    &ldquo;Copilot&rdquo; was removed from <code>master_keywords.json</code> on 2026-07-30
    (aviation false-positive collisions), so &ldquo;Copilot matched&rdquo; can never be true
    today. Genuine Copilot mentions are still caught via the unambiguous &ldquo;GitHub
    Copilot&rdquo;/&ldquo;Microsoft Copilot&rdquo; compound G1 keywords, just not through Rule C
    specifically.</p>
    """

    param_rows = [
        ("trend_start_year", "2016", "Fixed", "First pre-period year."),
        ("trend_end_year", "2020", "Fixed", "Last pre-period year (2021 excluded, see above)."),
        ("validation_year", "2025", "Fixed", "The year compared against the benchmark."),
        ("min_validation_hits_g2", "10", "Fixed", "Required within a single language &mdash; stricter than 10 summed across all languages."),
        ("min_validation_hits_g1", "1", "<span class='tag open'>OPEN</span>", f"Minimum for G1 to mean anything, summed across all languages AND across {G1_VALIDATION_YEARS[0]}&ndash;{G1_VALIDATION_YEARS[-1]} (changed 2026-08-06, was 2025 alone). Marked open in Jeremias's spec, unlike the other Fixed rows."),
        ("multiple_g2_threshold (c)", ", ".join(f"{c:g}" for c in C_GRID), "<span class='tag calibrate'>CALIBRATE</span>", "Grid tested by this dashboard's sweep."),
        ("min_keywords_g2 (k)", ", ".join(str(k) for k in K_GRID), "<span class='tag calibrate'>CALIBRATE</span>", "Grid tested by this dashboard's sweep."),
    ]
    param_table = "".join(
        f"<tr><td><code>{name}</code></td><td>{val}</td><td>{status}</td><td>{note}</td></tr>"
        for name, val, status, note in param_rows
    )

    return f"""
    {strategy_para}
    <h3>Notation</h3>
    <table class="datatable notation"><tr><th>Symbol</th><th>Meaning</th></tr>{notation_table}</table>
    {steps}
    <h3>Parameters</h3>
    <table class="datatable">
    <tr><th>Name</th><th>Value(s)</th><th>Status</th><th>Note</th></tr>
    {param_table}
    </table>
    <p style="color:#666;font-size:13px">c and k are both TBD &mdash; calibrated primarily via
    a hand-labelled precision sample (not yet done), with the grid sweep below as
    a secondary stability check only (where the curve flattens is not itself evidence of
    correctness &mdash; the bend can fall anywhere, including where precision is poor).</p>
    """


def main():
    parser = argparse.ArgumentParser(description="Build the v7 scoring results dashboard")
    parser.add_argument("--config", type=str, default="pipeline/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    master_keywords = load_master_keywords()
    all_kws_full = [k["keyword"] for k in master_keywords if k["keyword"] not in MANUALLY_EXCLUDED]

    total_2025_ads = (
        pl.scan_parquet(Path(config.output_dir) / "classified_data" / "year=*" / "*.parquet")
        .filter(pl.col("tst_created").dt.year() == VALIDATION_YEAR)
        .select(pl.len()).collect().item()
    )

    print("Run 1/2: full keyword pool...")
    en_p, local_p = build_keyword_patterns(master_keywords)
    counts_full, matches_full = collect_matches(config, en_p, local_p, MANUALLY_EXCLUDED)
    stats_full = classify_keywords(counts_full, all_kws_full)
    sweep_full = run_grid_sweep(stats_full, matches_full, total_2025_ads)
    n_g1_full = sum(1 for s in stats_full.values() if s["is_g1"])

    print("Run 2/2: Layer 3 excluded...")
    excluded_kws = {k["keyword"] for k in master_keywords if k["source"] == "LLM"} | MANUALLY_EXCLUDED
    all_kws_nl3 = [k for k in all_kws_full if k not in excluded_kws]
    en_p2, local_p2 = build_keyword_patterns(master_keywords)
    counts_nl3, matches_nl3 = collect_matches(config, en_p2, local_p2, excluded_kws)
    stats_nl3 = classify_keywords(counts_nl3, all_kws_nl3)
    sweep_nl3 = run_grid_sweep(stats_nl3, matches_nl3, total_2025_ads)
    n_g1_nl3 = sum(1 for s in stats_nl3.values() if s["is_g1"])

    # --- Assemble HTML ---
    print("Rendering LaTeX-style formulas and assembling HTML...")
    sections = []

    sections.append(("Methodology", "html", build_methodology_section()))

    g2_full = g2_keywords_at_c(stats_full, EXAMPLE_C)
    g2_nl3 = g2_keywords_at_c(stats_nl3, EXAMPLE_C)
    summary_full = _kw_summary_df(stats_full, g2_full)
    summary_nl3 = _kw_summary_df(stats_nl3, g2_nl3)

    sections.append((
        f"Elbow plot &mdash; full keyword pool ({n_g1_full}/{len(all_kws_full)} keywords are G1)",
        "html",
        pio.to_html(chart_elbow(sweep_full, "% of 2025 ads flagged vs. k (full keyword pool)"),
                    full_html=False, include_plotlyjs="inline")
        + elbow_flatten_note(sweep_full)
    ))
    sections.append((
        f"Elbow plot &mdash; Layer 3 excluded ({n_g1_nl3}/{len(all_kws_nl3)} keywords are G1, "
        f"{len(excluded_kws)} Layer-3 keywords removed)",
        "html",
        pio.to_html(chart_elbow(sweep_nl3, "% of 2025 ads flagged vs. k (Layer 3 excluded)"),
                    full_html=False, include_plotlyjs=False)
        + elbow_flatten_note(sweep_nl3)
    ))

    sections.append(("Full (c, k) landscape &mdash; c on x-axis", "html",
        "<p>The mirror image of the elbow plots above: c on the x-axis, one line per k. With "
        "only 49 keywords this axis barely moved (too few G2 candidates for c to bite on), so "
        f"it was dropped in favor of the k-axis view. With {len(all_kws_full)} keywords - a much "
        "bigger and more spread-out G2 candidate pool - c now has real effect worth seeing "
        "directly.</p>"
        + pio.to_html(chart_c_elbow(sweep_full, "% of 2025 ads flagged vs. c, one line per k (full keyword pool)"),
                       full_html=False, include_plotlyjs=False)
        + pio.to_html(chart_c_elbow(sweep_nl3, "% of 2025 ads flagged vs. c, one line per k (Layer 3 excluded)"),
                       full_html=False, include_plotlyjs=False)
    ))

    c_sensitivity_html = (
        f"<p>k=2 and k=3 are the two values under active discussion - these fix each in turn "
        f"and isolate what c alone does, without the other k-lines in the way.</p>"
    )
    for k in CANDIDATE_KS:
        c_sensitivity_html += (
            f"<h3>k={k}</h3>"
            + pio.to_html(
                chart_c_sensitivity(sweep_full, k, f"% of 2025 ads flagged vs. c, at k={k} (full keyword pool)"),
                full_html=False, include_plotlyjs=False
            )
            + pio.to_html(
                chart_c_sensitivity(sweep_nl3, k, f"% of 2025 ads flagged vs. c, at k={k} (Layer 3 excluded)"),
                full_html=False, include_plotlyjs=False
            )
        )
    sections.append(("C sensitivity, k=2 vs. k=3", "html", c_sensitivity_html))

    analysis_2 = newly_swept_in_analysis(stats_full, matches_full, 2)
    analysis_3 = newly_swept_in_analysis(stats_full, matches_full, 3)

    def _driver_rows_html(drivers, denom):
        if not drivers:
            return "<p><i>No newly-swept-in ads at this k.</i></p>"
        rows = "".join(
            f"<tr><td>{kw}</td><td>{n}</td><td>{n/denom*100:.0f}%</td></tr>"
            for kw, n in drivers[:8]
        )
        return (
            "<table class='datatable'><tr><th>Keyword</th><th># ads</th>"
            "<th>% of newly-swept-in ads mentioning it</th></tr>" + rows + "</table>"
        )

    top2 = analysis_2["top_driver"]
    top2_share = (top2[1] / analysis_2["newly_swept_in"] * 100) if top2 and analysis_2["newly_swept_in"] else 0
    dominance_flag = (
        f"<p style='color:#a15c00'><b>Same red flag as the pre-v7 system's Option B analysis "
        f"(where k=2's newly-flagged ads were 58% driven by MLOps alone):</b> at k=2, "
        f"&ldquo;{top2[0]}&rdquo; alone appears in {top2_share:.0f}% of the "
        f"{analysis_2['newly_swept_in']} newly-swept-in ads. That's not really independent "
        f"multi-signal corroboration - it's mostly &ldquo;{top2[0]} plus one loosely-related "
        f"term,&rdquo; a weaker case than genuinely distinct GenAI signals appearing "
        f"together.</p>" if top2 and top2_share >= 50 else ""
    )

    # Both figures below are computed fresh from analysis_2/analysis_3 every
    # run (2026-08-06 fix) - this section used to hardcode "~3% difference"
    # and "k=3 looks like the safer choice" as static prose, which silently
    # went stale the moment c/the keyword pool changed (the numbers in the
    # table above them were still live, but the sentences describing them
    # weren't - exactly the kind of mismatch that isn't safe to leave sitting
    # next to real data).
    k2_total, k3_total = analysis_2["flagged_total"], analysis_3["flagged_total"]
    pct_diff = abs(k2_total - k3_total) / k2_total * 100 if k2_total else 0.0
    reading_para = (
        f"<p><b>Our reading, for context - not a directive:</b> k=2 flags {k2_total:,}, k=3 "
        f"flags {k3_total:,} ({pct_diff:.1f}% difference). "
        + (
            f"k=2's newly-swept-in ads are dominated by a single keyword (&ldquo;{top2[0]}&rdquo; "
            f"at {top2_share:.0f}%, flagged above) - that's the concrete reason to lean toward k=3 "
            f"here, not just the size of the gap."
            if top2 and top2_share >= 50 else
            "No single-keyword dominance issue showed up for k=2 this run - the case for k=3 over "
            "k=2 is weaker than in earlier runs where it did."
        )
        + " This is still provisional pending the hand-labelled precision sample - included here "
        "so the full picture is available to decide from, not to settle it in advance.</p>"
    )

    sections.append((f"Choosing k: k=2 vs. k=3 (full keyword pool, c={EXAMPLE_C:g})", "html", f"""
    <p>Comparison baseline is <b>G1-alone</b> (an ad flagged just by
    {latex('N_{G1} \\geq 1')}, ignoring G2 entirely) - not k=1, since k=1 already accepts any
    single G2 match on its own and is therefore the loosest possible threshold, always a
    superset of every larger k.</p>
    <table class="datatable">
    <tr><th></th><th>Total flagged</th><th>Newly swept in beyond G1-alone</th></tr>
    <tr><td>G1-alone</td><td>{analysis_2['g1_alone_total']:,}</td><td>&mdash;</td></tr>
    <tr><td><b>k=2</b></td><td>{analysis_2['flagged_total']:,}</td><td>{analysis_2['newly_swept_in']:,}</td></tr>
    <tr><td><b>k=3</b></td><td>{analysis_3['flagged_total']:,}</td><td>{analysis_3['newly_swept_in']:,}</td></tr>
    </table>
    <p>k=2 vs. k=3 differ by {pct_diff:.1f}%.</p>
    <h3>k=2 newly-swept-in driver keywords</h3>
    {_driver_rows_html(analysis_2['drivers'], analysis_2['newly_swept_in'])}
    <h3>k=3 newly-swept-in driver keywords</h3>
    {_driver_rows_html(analysis_3['drivers'], analysis_3['newly_swept_in'])}
    {dominance_flag}
    {reading_para}
    """))

    sections.append(("G2 pool size vs. c", "html",
        "<p>How many keywords qualify as G2 at each candidate c - explains part of why the "
        "elbow plots and grid below look the way they do (fewer G2 keywords at higher c means "
        "fewer ads can reach a given k).</p>"
        + pio.to_html(chart_g2_count_vs_c(sweep_full, "G2 keyword count vs. c (full pool)"),
                       full_html=False, include_plotlyjs=False)
    ))

    sections.append(("Grid sweep - full (c, k) matrix", "html",
        "<p>Every <code>(c, k)</code> combination tested, both runs, as a plain c &times; k "
        "matrix - the exact numbers behind the elbow plots above, in a natural reading order "
        "(rows=c, columns=k) rather than a flat repeating list.</p>"
        "<h3>Full keyword pool</h3>" + pivot_table_html(sweep_full) +
        "<h3>Layer 3 excluded</h3>" + pivot_table_html(sweep_nl3)
    ))

    # Per Jeremias, 2026-08-06: he liked these tables and asked for more
    # columns - mean before 2020, y2025, G1 status, per-keyword multiplier
    # (already had this - it's breakeven_c below), how many ads it actually
    # helped flag, and y2023-2025 combined. Merged G1 + G2/neither into ONE
    # table (was two) so "is_g1" is a real column instead of an implicit
    # split, and added the new columns to the same summary_full frame.
    flagged_k2 = per_keyword_flagged_counts(stats_full, matches_full, 2)
    flagged_k3 = per_keyword_flagged_counts(stats_full, matches_full, 3)
    recent_volume = per_keyword_recent_volume(counts_full, all_kws_full)

    full_table = summary_full.copy()
    full_table["flagged_k2"] = full_table["keyword"].map(flagged_k2).fillna(0).astype(int)
    full_table["flagged_k3"] = full_table["keyword"].map(flagged_k3).fillna(0).astype(int)
    full_table["y2023_2025"] = full_table["keyword"].map(recent_volume).fillna(0).astype(int)

    # Per Louis, 2026-08-06 (the LLM/G1 confusion): show the ACTUAL matched
    # English text next to any keyword whose display name doesn't match it
    # (FORM_OVERRIDES cases - "LLM"/"RAG" today), blank everywhere else.
    en_form_by_kw = {k["keyword"]: k["forms"]["en"] for k in master_keywords}
    full_table["matched_text"] = full_table["keyword"].map(
        lambda kw: en_form_by_kw.get(kw, kw) if en_form_by_kw.get(kw, kw) != kw else ""
    )

    # Helper columns for full_keyword_table_html's grouping/sorting/highlight
    # logic - kept separate from the display columns (is_g1 stays Yes/No,
    # breakeven_c stays however _kw_summary_df formatted it: a rounded float,
    # "inf", or the G1 placeholder string) so the table still displays right.
    full_table["is_g1_bool"] = full_table["is_g1"]
    full_table["breakeven_c_num"] = full_table.apply(
        lambda r: float("inf") if r["is_g1"] or r["breakeven_c"] == "inf"
        else (float(r["breakeven_c"]) if r["breakeven_c"] is not None else float("inf")),
        axis=1,
    )
    full_table["is_g1"] = full_table["is_g1"].map({True: "Yes", False: "No"})

    n_all = len(full_table)
    sections.append(("Per-keyword classification (full pool)", "html",
        f"<p>Every one of the {n_all} keywords, one row each, in three groups: keywords with real "
        f"2025 signal (sorted by <code>breakeven_c</code> ascending, so the weakest/most-generic "
        f"candidates - easiest to exclude regardless of which c ends up chosen - are at the top), "
        f"then G1 keywords (immune to c), then keywords with zero 2025 matches at all - moved out "
        f"of the way at the bottom instead of sitting at breakeven_c=0 and crowding out the "
        f"genuinely borderline candidates above them.</p>"
        + pio.to_html(chart_keyword_ranking(summary_full, "Keywords ranked by 2025 volume", top_n=n_all),
                       full_html=False, include_plotlyjs=False)
        + "<p style='color:#666;font-size:13px'>"
          "<code>mean_pre2020</code>/<code>y2025_best_lang</code>/<code>B_best_lang</code>/"
          "<code>breakeven_c</code> are all for whichever language has this keyword's highest "
          "multiple - the one language that actually determines its G2 status, since G2 only "
          "needs ONE language to qualify. <code>breakeven_c</code> is the per-keyword multiplier: "
          "the exact c value (y2025 &divide; B) at which this keyword stops qualifying as G2 - it "
          "qualifies only while <code>c &le; breakeven_c</code>. <code>flagged_k2</code>/"
          "<code>flagged_k3</code> are how many 2025 ads this keyword matched that ALSO ended up "
          f"in the final ad-level flagged set (at c={EXAMPLE_C}) - lower than "
          "<code>y2025_all_langs</code> whenever the ad didn't have enough other corroborating "
          "keywords. <code>y2023_2025</code> is combined ad-match volume across the three most "
          "recent years, all languages - a sustained-volume check beyond the single 2025 "
          "snapshot. None of this captures the separate &ge;10-hits-in-one-language requirement, "
          "which can also be the binding constraint even when breakeven_c looks high. The "
        + f"{len(excluded_kws)} keywords with <code>source == \"LLM\"</code> (Layer 3 / "
          "layer3.py) are the ones excluded in the Layer-3 sensitivity run above.</p>"
        + full_keyword_table_html(full_table, max_c=max(C_GRID))
    ))

    def _nav_label(title: str) -> str:
        # Was: split on the first " &mdash;" and the first " (", whichever
        # came first - that mangled any title with a "(" that ISN'T a
        # trailing stats suffix (e.g. "Full (c, k) landscape &mdash; c on
        # x-axis" collapsed to just "Full"), and dropped every section-2/3-
        # style distinguishing suffix, leaving two identical "Elbow plot"
        # nav links with no way to tell them apart (2026-08-06 fix). Now:
        # only strip a trailing "(...)" stats suffix (regex-anchored to the
        # end, not wherever the first "(" happens to be), then swap the
        # &mdash; separator for a plain colon.
        t = re.sub(r"\s*\([^()]*\)$", "", title)
        return t.replace(" &mdash; ", ": ")

    nav_links = "".join(
        f'<a href="#s{i}">{i}. {_nav_label(title)}</a>'
        for i, (title, _kind, _payload) in enumerate(sections, start=1)
    )

    body_parts = [
        """<html><head><meta charset='utf-8'>
        <title>v7 Scoring Methodology - Results Dashboard</title>
        <style>
        :root{--ink:#1a1a2e;--muted:#666;--accent:#3b5bdb;--bg-soft:#f7f8fb;--border:#e1e4ea;}
        body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
             max-width:1000px;margin:0 auto;padding:24px 28px 80px;color:var(--ink);
             line-height:1.55;}
        h1{margin-bottom:2px;font-size:26px;}
        h2{font-size:19px;margin-top:0;border-bottom:2px solid var(--accent);padding-bottom:6px;}
        h3{font-size:15px;color:var(--ink);margin-top:26px;margin-bottom:8px;}
        .subtitle{color:var(--muted);margin-top:4px;margin-bottom:28px;font-size:14px;}
        .chart{margin-bottom:20px;padding:22px 26px;background:var(--bg-soft);
               border:1px solid var(--border);border-radius:10px;}
        hr{border:none;margin:34px 0;}
        p.lead{font-size:15px;background:#eef1fd;border-left:4px solid var(--accent);
               padding:12px 16px;border-radius:4px;}
        code{background:#eef0f4;padding:1px 5px;border-radius:4px;font-size:0.92em;}
        .tag{font-size:10.5px;font-weight:700;padding:2px 7px;border-radius:10px;
             letter-spacing:.03em;}
        .tag.calibrate{background:#fff3cd;color:#8a6400;}
        .tag.open{background:#ffe3e3;color:#a61e1e;}
        table.datatable{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 20px;}
        table.datatable th{background:#eef0f4;text-align:left;padding:7px 8px;
                           border-bottom:2px solid var(--border);}
        table.datatable td{padding:5px 8px;border-top:1px solid #eee;vertical-align:top;}
        table.notation td.sym{white-space:nowrap;text-align:center;padding:8px 14px;}
        nav.toc{position:sticky;top:0;background:#fff;padding:10px 0;margin-bottom:10px;
                border-bottom:1px solid var(--border);font-size:13px;z-index:10;}
        nav.toc a{margin-right:16px;color:var(--accent);text-decoration:none;}
        nav.toc a:hover{text-decoration:underline;}
        .scroll-table{max-height:640px;overflow-y:auto;border:1px solid var(--border);
                      border-radius:6px;margin:10px 0 20px;}
        .scroll-table table.datatable{margin:0;}
        .scroll-table thead th{position:sticky;top:0;z-index:5;box-shadow:0 1px 0 var(--border);}
        tr.group-break td{background:#eef0f4;font-weight:600;padding:7px 8px;color:var(--ink);
                          position:sticky;top:30px;z-index:4;}
        tr.would-cut td{background:#fff3cd;}
        </style></head><body>""",
        "<h1>v7 Scoring Methodology &mdash; Results Dashboard</h1>",
        f"<p class='subtitle'>Per Jeremias's spec (<code>genai_scoring_rules_v7.xlsx</code>, "
        f"2026-08-03). Computed on the full Dec 2016&ndash;Nov 2025 dataset "
        f"({total_2025_ads:,} 2025 ads). <b>Not a final calibrated result</b> &mdash; c and k "
        f"are still TBD, pending the hand-labelled sample. Updated 2026-08-06 per Jeremias: G1's "
        f"validation check now uses {G1_VALIDATION_YEARS[0]}&ndash;{G1_VALIDATION_YEARS[-1]} "
        f"combined (was 2025 alone), c grid raised to "
        f"{', '.join(f'{c:g}' for c in C_GRID)} (was 1.05&ndash;1.50), and "
        f"&ldquo;{'&rdquo;, &ldquo;'.join(sorted(MANUALLY_EXCLUDED))}&rdquo; are dropped from "
        f"consideration entirely. See "
        f"<code>notes/phase_b_v7_scoring_2026-08-03.md</code> for the earlier narrative.</p>",
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
    out_path = out_dir / "v7_dashboard.html"
    out_path.write_text("\n".join(body_parts), encoding="utf-8")
    print(f"\nSaved v7 dashboard to {out_path.resolve()}")


if __name__ == "__main__":
    main()
