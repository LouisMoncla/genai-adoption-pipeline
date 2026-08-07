"""
Layer-breakdown corroboration dashboard, per Jeremias's 2026-08-05 message
(relayed via a separate context/relay conversation, and confirmed directly
in his own words). He wants to know, at each candidate k, whether the ads
being flagged via keyword corroboration (i.e. NOT already flagged by a G1
keyword on its own) are being carried by solid, well-established keywords
or by the newer/shakier ones - broken into 4 lines:

  A - total # of 2025 ads flagged via corroboration (>=k distinct non-G1
      matched keywords, from the combined pool of everything below)
  B - of those, how many had >=1 contributing keyword from Layer 1+2
  C - of those, how many had >=1 contributing keyword from Layer 3a
  D - of those, how many had >=1 contributing keyword from Layer 3b

B/C/D are NOT a partition of A - they overlap by design (an ad flagged by
"Copilot" (3b) + "TensorFlow" (1+2) counts in both B and D). The point,
per Jeremias: "Copilot" alone is ambiguous, but "Copilot" + "TensorFlow" is
a much stronger signal - this shows whether the ambiguous 3b terms are
usually riding along on real corroboration, or pulling in noise alone.

IMPORTANT - "layer" here is NOT master_keywords.json's "group" field (1/2/3).
It's a different taxonomy, closer to the original keyword SOURCE:
  Layer 1 = source "St"  (Stanford list)
  Layer 2 = source "H&L" (Hosseini & Lin list)
  Layer 3 = source "LLM" (LLM-generated candidate list) - split into 3a/3b
This matches v7_scoring.py's existing --exclude-layer3 flag, which already
treats source=="LLM" as "Layer 3". See build_methodology_section() for the
full mapping, including Jeremias's explicit manual reassignments (moving
"Large language model"/"Microsoft Copilot" into 3a despite their source,
and "LLM"/"Foundation Model"/"Cohere" into 3b) and the 8 brand-new candidate
terms his 3b list includes that don't exist in master_keywords.json at all
yet (bare "Copilot", "T5", "Transformer", "Falcon", "Fine-Tuning", "Music
Generation", "Image Generation", "Jasper") - added here as English-only
experimental patterns, NOT written back to master_keywords.json.

ANOTHER READING CALLED OUT EXPLICITLY: Jeremias's message never mentions c
(the multiple_g2_threshold) at all - "any keyword that is not already in
G1" is read literally here as ANY non-G1 matched keyword, with no c/10-hit
filter. This is a real simplification relative to the main v7 dashboard's
G2 definition (which still requires clearing c*benchmark AND >=10 hits) -
flagged as our reading of his message, not a confirmed methodology change,
since he never says "drop c" in his own words.
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl

from pipeline.config_loader import load_config
from pipeline.group_classification import build_keyword_patterns, load_master_keywords, normalize

from v7_scoring import K_GRID, VALIDATION_YEAR, collect_matches, classify_keywords

# --- Layer taxonomy -----------------------------------------------------

# Existing master_keywords.json entries Jeremias explicitly moves INTO 3a,
# overriding their source-based default (both are source != "LLM" today).
FORCE_3A = {"Large language model", "Microsoft Copilot", "GitHub Copilot"}

# Existing entries Jeremias explicitly moves INTO 3b, overriding their
# source-based default.
FORCE_3B_EXISTING = {"LLM", "Foundation Model", "Cohere"}

# Brand-new candidate terms from Jeremias's 3b list that don't exist in
# master_keywords.json at all. English-only (no translated forms given) -
# flagged as a real limitation in the dashboard, not silently assumed
# complete. Also flagged: several of these are lexically risky standalone
# (bare "Transformer" collides with electrical transformers, "Falcon" with
# aircraft/brands, "Jasper" is a common personal name, "Copilot" was
# deliberately removed from the live system on 2026-07-30 for exactly this
# aviation-collision reason) - the corroboration check this whole dashboard
# is built around is partly meant to guard against exactly this risk.
NEW_3B_CANDIDATES = [
    "Copilot", "T5", "Transformer", "Falcon", "Fine-Tuning",
    "Music Generation", "Image Generation", "Jasper",
]

# Single combined regex to widen the candidate population (see
# v7_scoring.collect_matches's extra_candidate_regex) - applied to raw
# lowercased content_clean, so it only needs to be a superset of what the
# precise per-keyword normalized patterns below actually match.
EXTRA_CANDIDATE_REGEX = (
    r"copilot|\bt5\b|transformer|falcon|fine[-\s]?tuning"
    r"|music generation|image generation|jasper"
)


def keyword_layer(kw_entry: dict) -> str:
    name = kw_entry["keyword"]
    if name in FORCE_3A:
        return "3a"
    if name in FORCE_3B_EXISTING:
        return "3b"
    if kw_entry["source"] in ("St", "H&L"):
        return "1+2"
    if kw_entry["source"] == "LLM":
        return "3a"
    raise ValueError(f"Unexpected source for {name!r}: {kw_entry['source']!r}")


def build_extended_patterns(master_keywords: list[dict]):
    """Existing keyword patterns (unchanged) plus the 8 new-candidate
    English-only patterns, compiled the exact same way build_keyword_patterns
    does (normalize + escape + word-boundary) for consistent matching
    behavior."""
    en_patterns, local_patterns = build_keyword_patterns(master_keywords)
    for name in NEW_3B_CANDIDATES:
        norm = normalize(name)
        en_patterns.append((name, None, re.compile(r"\b" + re.escape(norm) + r"\b")))
    return en_patterns, local_patterns


def latex(formula: str, fontsize: int = 15, color: str = "#1a1a1a", display: bool = False) -> str:
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


def compute_layer_sweep(matches: list, g1_kws: set, layer_of: dict) -> pd.DataFrame:
    """For each k in K_GRID: A = # 2025 ads with >=k distinct non-G1 matched
    keywords; B/C/D = of those, # with >=1 contributing keyword from
    Layer 1+2 / 3a / 3b respectively. B/C/D overlap by design - see module
    docstring."""
    ads_2025 = [(ad_id, kws) for ad_id, year, kws in matches if year == VALIDATION_YEAR]
    non_g1_by_ad = {ad_id: [kw for kw in kws if kw not in g1_kws] for ad_id, kws in ads_2025}

    rows = []
    for k in K_GRID:
        a = b = c = d = 0
        for non_g1 in non_g1_by_ad.values():
            if len(non_g1) < k:
                continue
            a += 1
            layers_present = {layer_of[kw] for kw in non_g1}
            if "1+2" in layers_present:
                b += 1
            if "3a" in layers_present:
                c += 1
            if "3b" in layers_present:
                d += 1
        rows.append({"k": k, "A_total": a, "B_layer12": b, "C_layer3a": c, "D_layer3b": d})
    return pd.DataFrame(rows)


def chart_layer_elbow(df: pd.DataFrame, title: str):
    fig = go.Figure()
    series = [
        ("A_total", "A - Total flagged (any non-G1 corroboration)", "#1a1a1a", "solid"),
        ("B_layer12", "B - has a Layer 1+2 contributor", "#2ca02c", "dash"),
        ("C_layer3a", "C - has a Layer 3a contributor", "#1f77b4", "dash"),
        ("D_layer3b", "D - has a Layer 3b contributor", "#d62728", "dash"),
    ]
    for col, label, color, dash in series:
        fig.add_trace(go.Scatter(
            x=df["k"], y=df[col], mode="lines+markers+text", name=label,
            line=dict(color=color, width=3 if col == "A_total" else 2, dash=dash),
            marker=dict(size=9), text=df[col], textposition="top center",
        ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="k  (min. distinct non-G1 keywords required)", dtick=1),
        yaxis=dict(title="# of 2025 ads"),
        legend_title_text="",
    )
    return fig


def driver_table_html(matches: list, g1_kws: set, layer_of: dict, k: int, target_layer: str) -> str:
    """Which specific keywords from target_layer are actually firing among
    the k-flagged ads, and how often - the same "don't just trust the
    aggregate count" pattern used elsewhere in this project (see
    newly_swept_in_analysis in build_v7_dashboard.py)."""
    ads_2025 = [(ad_id, kws) for ad_id, year, kws in matches if year == VALIDATION_YEAR]
    counts: dict = {}
    n_flagged_with_layer = 0
    for _ad_id, kws in ads_2025:
        non_g1 = [kw for kw in kws if kw not in g1_kws]
        if len(non_g1) < k:
            continue
        layer_kws = [kw for kw in non_g1 if layer_of[kw] == target_layer]
        if not layer_kws:
            continue
        n_flagged_with_layer += 1
        for kw in layer_kws:
            counts[kw] = counts.get(kw, 0) + 1
    if not counts:
        return "<p><i>No ads at this k have a contributor from this layer.</i></p>"
    rows = "".join(
        f"<tr><td>{kw}</td><td>{n}</td><td>{n / n_flagged_with_layer * 100:.0f}%</td></tr>"
        for kw, n in sorted(counts.items(), key=lambda x: -x[1])
    )
    return (
        f"<table class='datatable'><tr><th>Keyword</th><th># ads</th>"
        f"<th>% of the {n_flagged_with_layer} ads this layer contributes to at k={k}</th></tr>"
        f"{rows}</table>"
    )


def keyword_assignment_table_html(master_keywords: list[dict], kw_stats: dict, layer_of: dict) -> str:
    rows = []
    for kw in master_keywords:
        name = kw["keyword"]
        is_g1 = kw_stats[name]["is_g1"]
        rows.append({
            "keyword": name, "master_source": kw["source"], "master_group": kw["group"],
            "this_analysis_layer": "G1 (excluded from sweep)" if is_g1 else layer_of[name],
            "y2025": kw_stats[name]["y2025_all_langs"],
            "new_candidate": "",
        })
    for name in NEW_3B_CANDIDATES:
        is_g1 = kw_stats[name]["is_g1"]
        rows.append({
            "keyword": name, "master_source": "-", "master_group": "-",
            "this_analysis_layer": "G1 (excluded from sweep)" if is_g1 else layer_of[name],
            "y2025": kw_stats[name]["y2025_all_langs"],
            "new_candidate": "NEW - English-only pattern",
        })
    df = pd.DataFrame(rows).sort_values(["this_analysis_layer", "y2025"], ascending=[True, False])
    return df.to_html(index=False, border=0, classes="datatable")


def build_methodology_section(n_new_g1: int) -> str:
    lx_k = latex("k")
    return f"""
    <p class="lead">Jeremias's question: once an ad is flagged via keyword <b>corroboration</b>
    (i.e. it has no G1 keyword on its own and needs {lx_k} distinct matching keywords to be
    trusted), which keywords are actually doing that work? This dashboard splits the
    non-G1 keyword pool into three tiers and, for each candidate {lx_k}, shows how many flagged
    ads had at least one contributor from each tier. The tiers are <b>not</b> mutually exclusive
    &mdash; an ad flagged by &ldquo;Copilot&rdquo; (Layer 3b) together with &ldquo;TensorFlow&rdquo;
    (Layer 1+2) counts toward <i>both</i> tiers, by design: the question isn&rsquo;t &ldquo;how many
    ads are flagged using <i>only</i> this tier,&rdquo; it&rsquo;s &ldquo;how many flagged ads have
    <i>this tier's</i> fingerprint on them at all.&rdquo; If Layer 3b's line stays low and mostly
    overlaps with Layer 1+2's, the ambiguous 3b terms are usually riding along on solid
    corroboration. If it doesn't, 3b may be pulling in noise on its own.</p>

    <h3>What &ldquo;layer&rdquo; means here (not the same as master_keywords.json's &ldquo;group&rdquo;)</h3>
    <p>Layer 1/2/3 tracks each keyword's original <b>source list</b>, not its current live
    <code>group</code> field (which is a separate, already-promoted classification and doesn't
    match this split): Layer 1 = Stanford list (<code>source == "St"</code>), Layer 2 = Hosseini
    &amp; Lin list (<code>source == "H&amp;L"</code>), Layer 3 = the LLM-generated candidate list
    (<code>source == "LLM"</code>) &mdash; the same definition <code>v7_scoring.py</code>'s
    <code>--exclude-layer3</code> flag already uses. Jeremias then hand-splits Layer 3 into:</p>
    <ul>
    <li><b>3a</b> &mdash; the less ambiguous terms (Dall&middot;E, Stable Diffusion, etc. &mdash;
    &ldquo;quite clearly linked to AI&rdquo;), plus two keywords manually moved in from
    elsewhere regardless of their source: <b>&ldquo;Large language model&rdquo;</b> and
    <b>&ldquo;Microsoft Copilot.&rdquo;</b></li>
    <li><b>3b</b> &mdash; the genuinely ambiguous ones. Three already exist in
    <code>master_keywords.json</code> and are moved here: <b>&ldquo;LLM&rdquo;</b> (bare
    abbreviation), <b>&ldquo;Foundation Model,&rdquo;</b> <b>&ldquo;Cohere.&rdquo;</b> The rest are
    <b>brand-new candidate terms that don't exist in the live keyword list at all</b>:
    bare &ldquo;Copilot,&rdquo; &ldquo;T5,&rdquo; &ldquo;Transformer,&rdquo; &ldquo;Falcon,&rdquo;
    &ldquo;Fine-Tuning,&rdquo; &ldquo;Music Generation,&rdquo; &ldquo;Image Generation,&rdquo;
    &ldquo;Jasper.&rdquo;</li>
    </ul>

    <p style="background:#fff3cd;border-left:4px solid #a15c00;padding:10px 14px;border-radius:4px;">
    <b>Two caveats on the 8 new candidate terms, worth reading before trusting Line D:</b><br>
    1. <b>English-only.</b> Jeremias gave no German/French/Italian forms, and guessing
    translations risks silently introducing wrong patterns &mdash; so these only match English
    ad text today. Non-English corroboration from this tier is undercounted.<br>
    2. <b>Several are lexically risky standalone words.</b> Bare &ldquo;Transformer&rdquo;
    collides with electrical transformers, &ldquo;Falcon&rdquo; with aircraft/brand names,
    &ldquo;Jasper&rdquo; is a common personal name, and bare &ldquo;Copilot&rdquo; is the exact
    term removed from the live system on 2026-07-30 for colliding with aviation co-pilot ads.
    {n_new_g1} of the 8 new terms came back G1 (zero 2016&ndash;2020, &ge;1 hit in 2025) rather
    than needing corroboration at all &mdash; worth a second look before trusting those blindly.
    This is genuinely part of what this chart is meant to surface, not just a disclaimer: if
    Line D tracks closely with B/C, the collision risk is likely being caught by the
    corroboration requirement anyway.</p>

    <p style="color:#666;font-size:13px"><b>One more reading worth flagging:</b> Jeremias's own
    message never mentions <code>c</code> (the multiple-of-benchmark threshold) &mdash;
    &ldquo;any keyword that is not already in G1&rdquo; is read here literally, as <i>any</i>
    non-G1 matched keyword, with no <code>c</code> or 10-hit filter applied. That's a real
    simplification relative to the main v7 dashboard's G2 definition, which still requires
    clearing <code>c &times; benchmark</code> <i>and</i> &ge;10 hits. Treated here as our best
    reading of his message, not a confirmed methodology change &mdash; worth confirming with him
    directly before this replaces the c-based G2 definition anywhere else.</p>
    """


def main():
    config = load_config("pipeline/config.yaml")
    master_keywords = load_master_keywords()

    layer_of = {kw["keyword"]: keyword_layer(kw) for kw in master_keywords}
    for name in NEW_3B_CANDIDATES:
        layer_of[name] = "3b"

    print("Matching keywords (existing 49 + 8 new Layer-3b candidates)...")
    en_patterns, local_patterns = build_extended_patterns(master_keywords)
    counts, matches = collect_matches(
        config, en_patterns, local_patterns, extra_candidate_regex=EXTRA_CANDIDATE_REGEX
    )

    all_kws = [k["keyword"] for k in master_keywords] + list(NEW_3B_CANDIDATES)
    print("Classifying keywords (G1 status)...")
    kw_stats = classify_keywords(counts, all_kws)
    g1_kws = {kw for kw, s in kw_stats.items() if s["is_g1"]}
    n_new_g1 = sum(1 for name in NEW_3B_CANDIDATES if kw_stats[name]["is_g1"])
    print(f"  {len(g1_kws)}/{len(all_kws)} keywords are G1 ({n_new_g1}/8 of the new candidates).")

    print("Running k-sweep with layer breakdown...")
    sweep_df = compute_layer_sweep(matches, g1_kws, layer_of)

    total_2025_ads = (
        pl.scan_parquet(Path(config.output_dir) / "classified_data" / "year=*" / "*.parquet")
        .filter(pl.col("tst_created").dt.year() == VALIDATION_YEAR)
        .select(pl.len()).collect().item()
    )

    print("Assembling dashboard...")
    sections = []
    sections.append(("Methodology", "html", build_methodology_section(n_new_g1)))

    sections.append(("A/B/C/D elbow plot", "figure", chart_layer_elbow(
        sweep_df, "Corroborated 2025 ads by contributing layer, vs. k"
    )))

    pct_df = sweep_df.copy()
    for col in ("B_layer12", "C_layer3a", "D_layer3b"):
        pct_df[col.split("_")[0] + "_pct_of_A"] = (
            100 * sweep_df[col] / sweep_df["A_total"].replace(0, pd.NA)
        ).round(1)
    table_cols = ["k", "A_total", "B_layer12", "C_layer3a", "D_layer3b",
                  "B_pct_of_A", "C_pct_of_A", "D_pct_of_A"]
    sections.append(("Numbers behind the plot", "html",
        "<p>B/C/D and their % columns are share of <b>A</b> (this k's total corroborated "
        "count) that has a contributor from that layer &mdash; not mutually exclusive, so "
        "rows don't sum to A.</p>"
        + pct_df[table_cols].to_html(index=False, border=0, classes="datatable")
    ))

    driver_html = ""
    for k in (2, 3):
        driver_html += f"<h3>k={k} &mdash; Layer 3b contributors</h3>" + driver_table_html(
            matches, g1_kws, layer_of, k, "3b"
        )
    sections.append(("Layer 3b drivers at k=2 and k=3", "html",
        "<p>Which specific Layer 3b keywords are actually firing among the corroborated ads "
        "at each k, and how often &mdash; the aggregate D line above can hide a single "
        "dominant (possibly noisy) term the same way MLOps dominated the pre-v7 k=2 "
        "analysis.</p>" + driver_html
    ))

    sections.append(("Keyword &rarr; layer assignment (full transparency)", "html",
        "<p>Every keyword's assigned layer for this analysis, its original "
        "<code>master_keywords.json</code> source/group (for comparison), G1 status, and 2025 "
        "volume. The 8 rows marked <b>NEW</b> don't exist in the live keyword list.</p>"
        + keyword_assignment_table_html(master_keywords, kw_stats, layer_of)
    ))

    nav_links = "".join(
        f'<a href="#s{i}">{i}. {title.split(" (")[0]}</a>'
        for i, (title, _kind, _payload) in enumerate(sections, start=1)
    )

    body_parts = [
        f"""<html><head><meta charset='utf-8'>
        <title>Layer Breakdown &mdash; Corroboration Dashboard</title>
        <style>
        :root{{--ink:#1a1a2e;--muted:#666;--accent:#3b5bdb;--bg-soft:#f7f8fb;--border:#e1e4ea;}}
        body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
             max-width:1000px;margin:0 auto;padding:24px 28px 80px;color:var(--ink);
             line-height:1.55;}}
        h1{{margin-bottom:2px;font-size:26px;}}
        h2{{font-size:19px;margin-top:0;border-bottom:2px solid var(--accent);padding-bottom:6px;}}
        h3{{font-size:15px;color:var(--ink);margin-top:26px;margin-bottom:8px;}}
        .subtitle{{color:var(--muted);margin-top:4px;margin-bottom:28px;font-size:14px;}}
        .chart{{margin-bottom:20px;padding:22px 26px;background:var(--bg-soft);
               border:1px solid var(--border);border-radius:10px;}}
        p.lead{{font-size:15px;background:#eef1fd;border-left:4px solid var(--accent);
               padding:12px 16px;border-radius:4px;}}
        code{{background:#eef0f4;padding:1px 5px;border-radius:4px;font-size:0.92em;}}
        table.datatable{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 20px;}}
        table.datatable th{{background:#eef0f4;text-align:left;padding:7px 8px;
                           border-bottom:2px solid var(--border);}}
        table.datatable td{{padding:5px 8px;border-top:1px solid #eee;vertical-align:top;}}
        nav.toc{{position:sticky;top:0;background:#fff;padding:10px 0;margin-bottom:10px;
                border-bottom:1px solid var(--border);font-size:13px;z-index:10;}}
        nav.toc a{{margin-right:16px;color:var(--accent);text-decoration:none;}}
        nav.toc a:hover{{text-decoration:underline;}}
        </style></head><body>""",
        "<h1>Layer Breakdown &mdash; Corroboration Dashboard</h1>",
        f"<p class='subtitle'>Per Jeremias's message, 2026-08-05. Computed on the full Dec "
        f"2016&ndash;Nov 2025 dataset ({total_2025_ads:,} 2025 ads). Exploratory &mdash; the 8 "
        f"new Layer 3b candidate terms are experimental patterns added only for this analysis, "
        f"<b>not</b> written back to <code>master_keywords.json</code> or the live pipeline.</p>",
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
    out_path = out_dir / "layer_breakdown_dashboard.html"
    out_path.write_text("\n".join(body_parts), encoding="utf-8")
    print(f"\nSaved layer breakdown dashboard to {out_path.resolve()}")


if __name__ == "__main__":
    main()
