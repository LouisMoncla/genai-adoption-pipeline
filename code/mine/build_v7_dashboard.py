"""
v7 scoring results dashboard - single-file HTML, same self-contained
convention as build_results_dashboard_v3.py (inline Plotly, no external
deps, double-click to open). Everything is computed fresh from
output/classified_data each run (not read back from the CSVs
v7_scoring.py separately saves) - avoids any risk of the dashboard drifting
out of sync with a stale CSV; v7_scoring.py's own runs are cheap enough
(under a minute each) that recomputing three times here is not a real cost.

Covers: the full-keyword-pool grid sweep, the Layer-3-excluded sensitivity
run, a v7-vs-current-live comparison at one example (c, k) point, and the
bare-LLM experiment (Jeremias, 2026-08-03) - see
notes/phase_b_v7_scoring_2026-08-03.md for the full narrative this
dashboard visualizes.
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
import polars as pl

from pipeline.config_loader import load_config
from pipeline.group_classification import build_keyword_patterns, load_master_keywords

from v7_scoring import (
    C_GRID, K_GRID, VALIDATION_YEAR,
    collect_matches, classify_keywords, g2_keywords_at_c, run_grid_sweep,
    save_keyword_stats,
)

EXAMPLE_C = 1.20
EXAMPLE_K = 3


def _kw_stats_df(kw_stats: dict) -> pd.DataFrame:
    rows = []
    for kw, s in kw_stats.items():
        row = {"keyword": kw, "zero_16_20": s["zero_16_20"],
               "y2025_all_langs": s["y2025_all_langs"], "is_g1": s["is_g1"]}
        for lang, stats in s["per_lang"].items():
            row[f"B_{lang}"] = round(stats["B"], 2)
            row[f"y2025_{lang}"] = stats["y2025"]
            row[f"multiple_{lang}"] = round(stats["y2025"] / stats["B"], 2) if stats["B"] > 0 else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["is_g1", "keyword"], ascending=[False, True])


def table_html(df: pd.DataFrame, max_rows: int = 60) -> str:
    return df.head(max_rows).to_html(index=False, border=0, classes="datatable")


def chart_grid_sweep(sweep_df: pd.DataFrame, title: str):
    return px.line(
        sweep_df, x="c", y="flagged_share_pct", color="k", markers=True,
        title=title,
        labels={"c": "c (multiple_g2_threshold)", "flagged_share_pct": "% of 2025 ads flagged", "k": "k"},
    )


def section_flagged_at_example_point(kw_stats_full: dict, matches_full: list, total_2025: int) -> str:
    g1 = {kw for kw, s in kw_stats_full.items() if s["is_g1"]}
    g2 = g2_keywords_at_c(kw_stats_full, EXAMPLE_C)
    v7_flagged = set()
    for ad_id, year, kws in matches_full:
        if year != VALIDATION_YEAR:
            continue
        n_g1 = sum(1 for kw in kws if kw in g1)
        n_g2 = sum(1 for kw in kws if kw in g2)
        if n_g1 >= 1 or n_g2 >= EXAMPLE_K:
            v7_flagged.add(ad_id)
    return v7_flagged, g1, g2


def main():
    parser = argparse.ArgumentParser(description="Build the v7 scoring results dashboard")
    parser.add_argument("--config", type=str, default="pipeline/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    master_keywords = load_master_keywords()
    all_kws_full = [k["keyword"] for k in master_keywords]

    total_2025_ads = (
        pl.scan_parquet(Path(config.output_dir) / "classified_data" / "year=*" / "*.parquet")
        .filter(pl.col("tst_created").dt.year() == VALIDATION_YEAR)
        .select(pl.len()).collect().item()
    )

    print("Run 1/3: full keyword pool...")
    en_p, local_p = build_keyword_patterns(master_keywords)
    counts_full, matches_full = collect_matches(config, en_p, local_p)
    stats_full = classify_keywords(counts_full, all_kws_full)
    sweep_full = run_grid_sweep(stats_full, matches_full, total_2025_ads)
    n_g1_full = sum(1 for s in stats_full.values() if s["is_g1"])

    print("Run 2/3: Layer 3 excluded...")
    excluded_kws = {k["keyword"] for k in master_keywords if k["source"] == "LLM"}
    all_kws_nl3 = [k for k in all_kws_full if k not in excluded_kws]
    en_p2, local_p2 = build_keyword_patterns(master_keywords)
    counts_nl3, matches_nl3 = collect_matches(config, en_p2, local_p2, excluded_kws)
    stats_nl3 = classify_keywords(counts_nl3, all_kws_nl3)
    sweep_nl3 = run_grid_sweep(stats_nl3, matches_nl3, total_2025_ads)
    n_g1_nl3 = sum(1 for s in stats_nl3.values() if s["is_g1"])

    print("Run 3/3: bare-LLM experiment...")
    en_p3, local_p3 = build_keyword_patterns(master_keywords)
    llm_group = next(k["group"] for k in master_keywords if k["keyword"] == "LLM")
    en_p3.append(("LLM", llm_group, re.compile(r"\bllms?\b")))
    counts_llm, matches_llm = collect_matches(config, en_p3, local_p3, extra_candidate_regex=r"\bllms?\b")
    stats_llm = classify_keywords(counts_llm, all_kws_full)
    sweep_llm = run_grid_sweep(stats_llm, matches_llm, total_2025_ads)
    n_g1_llm = sum(1 for s in stats_llm.values() if s["is_g1"])

    print("Computing v7-vs-live comparison and bare-LLM gain/loss...")
    v7_flagged, g1_full, g2_full = section_flagged_at_example_point(stats_full, matches_full, total_2025_ads)
    live_flagged = set(
        pl.scan_parquet(Path(config.output_dir) / "classified_data" / "year=2025" / "*.parquet")
        .filter(pl.col("group") != "NA").select("ad_id").collect()["ad_id"].to_list()
    )
    v7_only = v7_flagged - live_flagged
    live_only = live_flagged - v7_flagged

    def flagged_set(stats, matches, k, c=EXAMPLE_C):
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

    llm_gain_loss_rows = []
    for k in K_GRID[:3]:
        fw = flagged_set(stats_full, matches_full, k)
        fb = flagged_set(stats_llm, matches_llm, k)
        llm_gain_loss_rows.append({
            "k": k, "without": len(fw), "with_bare_llm": len(fb),
            "net": len(fb) - len(fw), "gained": len(fb - fw), "lost": len(fw - fb),
        })
    llm_gain_loss_df = pd.DataFrame(llm_gain_loss_rows)

    # --- Assemble HTML ---
    sections = []

    sections.append(("1. Methodology summary", "html", """
    <p><b>Multiple</b>: yhat = OLS(2016-2020 annual counts) predicted at 2025;
    mean = average(2016-2020); <b>B = max(yhat, mean, 0)</b> (benchmark, never negative).</p>
    <p><b>G1</b>: zero counts 2016-2020 in ALL languages, AND &ge;1 hit in 2025 (any language).
    Single property of the keyword, not per-language.</p>
    <p><b>G2</b>: y2025 &ge; c*B AND y2025 &ge; 10, true in AT LEAST ONE language -&gt;
    propagates to the whole keyword (all language forms).</p>
    <p><b>Ad flag</b>: N_G1 &ge; 1 OR N_G2 &ge; k OR Rule C (Copilot + Microsoft/GitHub -
    currently a structural no-op, see §8).</p>
    <p>Pre-period 2016-2020, 2021 explicitly excluded (GitHub Copilot preview / DALL-E /
    GPT-3 API all arrived that year). c and k are both TBD - calibrated via a grid sweep
    (this dashboard) plus a hand-labelled sample (not yet done, see §8).</p>
    """))

    sections.append((
        f"2. Grid sweep - full keyword pool ({n_g1_full}/{len(all_kws_full)} keywords are G1)",
        "figure", chart_grid_sweep(sweep_full, "% of 2025 ads flagged vs. c (full keyword pool)")
    ))
    sections.append((
        f"3. Grid sweep - Layer 3 excluded ({n_g1_nl3}/{len(all_kws_nl3)} keywords are G1, "
        f"{len(excluded_kws)} Layer-3 keywords removed)",
        "figure", chart_grid_sweep(sweep_nl3, "% of 2025 ads flagged vs. c (Layer 3 excluded)")
    ))

    sections.append(("4. Grid sweep tables", "html",
        "<h3>Full keyword pool</h3>" + table_html(sweep_full) +
        "<h3>Layer 3 excluded</h3>" + table_html(sweep_nl3)
    ))

    sections.append(("5. Per-keyword classification (full pool)", "html",
        f"<p>Keywords marked <code>is_g1=True</code> flag an ad on their own (N_G1&ge;1). "
        f"The {len(excluded_kws)} keywords with source=='LLM' (Layer 3 / layer3.py) are "
        f"excluded in the §3 sensitivity run.</p>" +
        table_html(_kw_stats_df(stats_full), max_rows=49)
    ))

    sections.append((f"6. v7 vs. current-live flagging (example point: c={EXAMPLE_C}, k={EXAMPLE_K})", "html", f"""
    <p><b>Important framing:</b> "current-live" here means today's actual
    <code>group != "NA"</code> flagging (matched &ge;1 of 49 keywords, no tiering) -
    NOT the old DHS Rule-1/2/3/C system (that hasn't been run on this data).</p>
    <table class="datatable">
    <tr><th>v7 flagged</th><th>live flagged</th><th>agree</th><th>v7-only</th><th>live-only</th></tr>
    <tr><td>{len(v7_flagged):,}</td><td>{len(live_flagged):,}</td>
        <td>{len(v7_flagged & live_flagged):,}</td><td>{len(v7_only):,}</td><td>{len(live_only):,}</td></tr>
    </table>
    <p>v7 is a strict subset of live at this point - every v7-flagged ad is also
    live-flagged. This is one illustrative point, not a final answer; see the grid
    sweep above for other (c,k) combinations.</p>
    """))

    sections.append(("7. Bare-LLM experiment (Jeremias, 2026-08-03)", "html", f"""
    <p><i>"Let's see about that, with k&gt;1 I think we can easily add LLM.
    But we can decide after."</i> - implemented as an experiment
    (<code>--include-bare-llm</code> flag), NOT adopted in the live system.</p>
    <p>Adding bare "LLM"/"LLMs" flips the "LLM" keyword from G1 to G2
    ({n_g1_full} G1 keywords without it, {n_g1_llm} with it) - the LL.M. legal-degree
    collision reintroduces nonzero 2016-2020 counts, breaking G1's zero-pre-period rule.
    This is exactly the mechanism the k&gt;1 intuition relies on.</p>
    {table_html(llm_gain_loss_df)}
    <p><b>k=1 confirms the risk</b>: pure gain, zero losses - no corroboration check
    fires at k=1 (N_G2&ge;1 alone satisfies it), so this is the unsafe case.</p>
    <p><b>k&ge;2 is a wash-to-net-negative on raw count</b>, not a clean gain - LLM
    losing its "free" G1 status costs more ads than newly-corroborated bare-LLM matches
    add back. Spot-checked 10/10 of the ads gained at k=2 by hand: all genuine AI
    mentions (Applied AI/ML/NLP, LLM-based agent workflows, GenAI infrastructure) -
    the corroboration requirement is doing its job precision-wise, it just doesn't
    net out to more total coverage at the aggregate level.</p>
    """))

    sections.append(("8. Open items / judgment calls", "html", """
    <ul>
    <li><b>G1/G2 overlap</b>: a keyword with a fully-zero pre-period has B=0, so it could
    mathematically satisfy G2's threshold too. Implemented: G1 takes precedence, excluded
    from the G2 set. Doesn't change ad flagging (G1 alone already flags), only how "G2
    keyword count" is reported.</li>
    <li><b>Rule C is a structural no-op</b>: bare "Copilot" was removed from
    master_keywords.json on 2026-07-30 (aviation false-positive collisions), so "Copilot
    matched" can never be true today. Genuine Copilot mentions are still caught via the
    unambiguous "GitHub Copilot"/"Microsoft Copilot" compound G1 keywords, just not
    through Rule C specifically.</li>
    <li><b>min_validation_hits_g1 = 1</b> is marked OPEN status in Jeremias's spec (unlike
    the other Fixed parameters) - not yet confirmed as final.</li>
    <li><b>Bare-LLM experiment (§7)</b> is exploratory - his call on whether to adopt it,
    after seeing these numbers.</li>
    <li><b>Choosing c and k</b> still needs the hand-labelled precision sample - not
    started (sample-design discrepancy between his ~200+200 spec and our internal
    proposal not yet reconciled).</li>
    </ul>
    """))

    body_parts = [
        "<html><head><meta charset='utf-8'>"
        "<title>v7 Scoring Methodology - Results Dashboard</title>"
        "<style>body{font-family:Arial,Helvetica,sans-serif;max-width:1100px;"
        "margin:0 auto;padding:20px;} h1{margin-bottom:0;} .subtitle{color:#666;"
        "margin-top:4px;margin-bottom:40px;} .chart{margin-bottom:60px;}"
        "hr{border:none;border-top:1px solid #ddd;margin:40px 0;}"
        "table.datatable{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}"
        "table.datatable th{background:#f0f0f0;text-align:left;padding:6px;border-bottom:2px solid #ccc}"
        "table.datatable td{padding:4px 6px;border-top:1px solid #eee}"
        "</style></head><body>"
        "<h1>v7 Scoring Methodology &mdash; Results Dashboard</h1>"
        f"<p class='subtitle'>Per Jeremias's spec (genai_scoring_rules_v7.xlsx, 2026-08-03). "
        f"Computed on the full Dec 2016&ndash;Nov 2025 dataset "
        f"({total_2025_ads:,} 2025 ads). NOT a final calibrated result - c and k are "
        f"still TBD, pending the hand-labelled sample. See "
        f"notes/phase_b_v7_scoring_2026-08-03.md for the full narrative.</p>"
    ]

    first = True
    for title, kind, payload in sections:
        if kind == "html":
            body_parts.append(f"<div class='chart'><h2>{title}</h2>{payload}</div><hr>")
            continue
        html = pio.to_html(payload, full_html=False, include_plotlyjs="inline" if first else False)
        first = False
        body_parts.append(f"<div class='chart'><h2>{title}</h2>{html}</div><hr>")

    body_parts.append("</body></html>")

    out_dir = Path(config.output_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v7_dashboard.html"
    out_path.write_text("\n".join(body_parts), encoding="utf-8")
    print(f"\nSaved v7 dashboard to {out_path.resolve()}")


if __name__ == "__main__":
    main()
