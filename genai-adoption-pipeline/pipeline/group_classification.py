"""
Group-priority classification using Domenico's actual validated keyword list
(keyword_lists/master_keywords.json, built by code/mine/build_master_keywords.py
from his thesis Table 4) instead of the placeholder flag+score approach in
simple_keyword_scoring.py.

WHAT THIS PRODUCES, per ad: two new columns.
  - ad_id : the ad's existing job-id column (config.col.job_id), carried through
            under a clearer name for this output.
  - group : "1", "2", or "3" - the HIGHEST-PRIORITY group of any keyword matched
            in the ad's text (Group 1 beats Group 2 beats Group 3), or "NA" if no
            validated keyword matched at all.

DELIBERATELY SIMPLER than Domenico's full flagging algorithm (2026-07-22 decision):
this is keyword-group matching only. It does NOT implement:
  - Rule 2's occupation-intensity gate (Appendix B's GenAI-intensive occupation list)
  - Rule 3's co-occurrence requirement (>=3 combined G2+G3 keywords)
Do not add these back in without checking first - explicitly asked for.

COPILOT: bare "Copilot" used to be a validated Group-2 keyword here, patched with an
aviation-context word-list exclusion after it drove a false-positive spike in the
occupation-share plot (co-pilot job ads). Per Jeremias (2026-07-30), that keyword is
now excluded entirely at merge time (see code/mine/build_master_keywords.py's
EXCLUDED_KEYWORDS) - genuine Copilot mentions are still caught via the separate,
unambiguous "GitHub Copilot"/"Microsoft Copilot" Group-1 keywords, so no in-code
exclusion logic is needed here anymore.

NORMALIZATION SPEC (from Jeremias, applied identically to both the ad text and every
keyword form before matching - see normalize()):
  - German digraph convention: u"->ue, o"->oe, a"->ae, ss->ss, so "Buero"/"buero"/"Buro"
    style spelling variants all collapse to one form.
  - Romance accents drop to their base letter (e/e`/a`/i`/o`... -> e/e/a/i/o) via Unicode
    NFKD decomposition + stripping combining marks - a general rule, not a hardcoded
    per-character table, so it covers accented letters beyond just French/Italian.
  - Hyphens and spaces are interchangeable: both collapse to a single space, so
    "Vector Database" and "Vector-Database" become identical strings.
  - Apostrophes and other punctuation are stripped entirely (only a-z, 0-9, and
    single spaces remain).
Matching is exact-phrase (word-boundary regex) on the normalized text - no fuzzy/typo
tolerance beyond what the normalization above already buys, per Jeremias's exact spec
(this replaced an earlier "use your judgment on fuzzy tolerance" instruction).

MATCHING SCOPE, per ad: its own detected-language keyword forms (from
language_detection.py's detected_language column) PLUS the English form of every
keyword (brand names appear verbatim regardless of ad language, per the thesis).
Ads with no local-language form available (e.g. detected_language == "rm", not in
master_keywords.json) are checked against English forms only - same convention
language_detection.py / config.yaml already documents for Romansh.

WHY THIS IS A SEPARATE MODULE FROM simple_keyword_scoring.py: that module answers a
different question (flag+score against ALL layer1/2/3 keywords, no tiering, built
2026-07-17 as a placeholder before the validated list existed). This module answers
Jeremias's actual current ask (group-priority classification against the VALIDATED
50-keyword list only). Both can coexist; simple_keyword_scoring.py is left as-is.

PARALLELIZED ACROSS FILES (added 2026-07-23): the per-row Python matching loop is
CPU-bound, not memory-bound - unlike the export/Phase I work earlier, there's no
benefit to a bigger memory cap here, only to more CPU cores. A single-threaded first
run projected to ~4 hours for the full 2020-2025 dataset based on the smallest year's
actual timing. Switched to multiprocessing.Pool across files (this machine has 12
logical cores and, per the user, is free to use all of them right now) - each worker
rebuilds the small (50-keyword) pattern set once via an initializer rather than
re-compiling per file. Also added a skip-if-already-classified check per file, so a
partial run (or one interrupted to make this exact change) resumes cheaply instead of
redoing finished files.
"""

import json
import logging
import multiprocessing
import re
import unicodedata
from pathlib import Path

import polars as pl

from pipeline.config_loader import PipelineConfig

logger = logging.getLogger(__name__)

N_WORKERS = 10  # of 12 logical cores - leaves a couple free for the OS/this script

MASTER_KEYWORDS_PATH = (
    Path(__file__).resolve().parent.parent / "keyword_lists" / "master_keywords.json"
)

_GERMAN_DIGRAPHS = str.maketrans({"ü": "ue", "ö": "oe", "ä": "ae", "ß": "ss"})


def normalize(text: str) -> str:
    """Fold spelling/accent variation so ad text and keyword forms can match
    regardless of how umlauts/accents/hyphenation were typed. See module
    docstring for the exact rules (from Jeremias)."""
    if not text:
        return ""
    t = text.lower()
    t = t.translate(_GERMAN_DIGRAPHS)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[-\s]+", " ", t)          # hyphens/spaces interchangeable
    t = re.sub(r"[^a-z0-9 ]", "", t)       # strip apostrophes/other punctuation
    return t.strip()


def load_master_keywords() -> list[dict]:
    data = json.loads(MASTER_KEYWORDS_PATH.read_text(encoding="utf-8"))
    return data["keywords"]


def build_keyword_patterns(master_keywords: list[dict]):
    """Precompile one word-boundary regex per keyword form, once, so the
    per-row loop below only ever calls .search() - never re.compile()."""
    en_patterns = []
    local_patterns = {"de": [], "fr": [], "it": []}

    for kw in master_keywords:
        forms = kw["forms"]
        en_norm = normalize(forms["en"])
        if en_norm:
            en_patterns.append(
                (kw["keyword"], kw["group"], re.compile(r"\b" + re.escape(en_norm) + r"\b"))
            )
        for lang in ("de", "fr", "it"):
            local_norm = normalize(forms.get(lang, ""))
            if local_norm and local_norm != en_norm:
                local_patterns[lang].append(
                    (kw["keyword"], kw["group"], re.compile(r"\b" + re.escape(local_norm) + r"\b"))
                )

    return en_patterns, local_patterns


def classify_text(normalized_text: str, lang: str, en_patterns, local_patterns) -> dict[str, int]:
    """Return {matched_keyword: group} for every validated keyword found in
    this (already-normalized) text."""
    matched: dict[str, int] = {}
    for kw, grp, pat in en_patterns:
        if pat.search(normalized_text):
            matched[kw] = grp
    for kw, grp, pat in local_patterns.get(lang, []):
        if pat.search(normalized_text):
            matched[kw] = grp
    return matched


_worker_patterns = None  # set once per worker process by _init_worker()


def _init_worker():
    """Pool initializer - runs once per worker process (not once per file),
    so the 50-keyword pattern set is only ever compiled N_WORKERS times, not
    once per file."""
    global _worker_patterns
    _worker_patterns = build_keyword_patterns(load_master_keywords())


def _classify_file_worker(args) -> tuple[str, int]:
    """Runs in a worker process. Must be a top-level function (not a
    closure/method) so multiprocessing can pickle it for Windows' spawn-based
    process start method."""
    pf, out_path, content_col, job_id_col = args
    en_patterns, local_patterns = _worker_patterns

    df = pl.read_parquet(pf)
    if df.is_empty():
        return (pf.name, 0)

    struct_series = df.select(
        pl.struct([content_col, "detected_language", job_id_col])
    ).to_series()

    ad_ids: list = []
    groups: list[str] = []
    trigger_lists: list[list[str]] = []

    for row in struct_series:
        text = row[content_col]
        lang = row["detected_language"]
        norm = normalize(text or "")

        matched = classify_text(norm, lang, en_patterns, local_patterns)
        if matched:
            best_group = min(matched.values())
            triggers = sorted(k for k, g in matched.items() if g == best_group)
            groups.append(str(best_group))
        else:
            triggers = []
            groups.append("NA")

        ad_ids.append(row[job_id_col])
        trigger_lists.append(triggers)

    df_out = df.with_columns([
        pl.Series("ad_id", ad_ids),
        pl.Series("group", groups),
        pl.Series("matched_group_keywords", trigger_lists, dtype=pl.List(pl.String)),
    ])
    # Write to a temp filename, rename only after a successful write - a kill
    # mid-write must never leave a truncated file at the real output path,
    # since the skip-if-exists check below only checks existence, not
    # validity. Found the hard way 2026-07-30: 5 files were left truncated by
    # a kill and silently passed the "already classified" skip check on the
    # next run, until scanning classified_data errored on them directly.
    # Mirrors code/mine/export_duckdb_to_parquet.py's existing pattern for the
    # same reason.
    tmp_path = out_path.with_suffix(".parquet.inprogress")
    df_out.write_parquet(tmp_path)
    tmp_path.replace(out_path)
    return (pf.name, df.height)


def classify_postings_by_group(config: PipelineConfig) -> None:
    """Entry point. Reads output/prepared_data (Phase I + language detection
    output), writes output/classified_data with ad_id + group added.
    Parallelized across files via a process pool - see module docstring."""
    output_dir = Path(config.output_dir)
    data_root = output_dir / "prepared_data"
    out_root = output_dir / "classified_data"
    out_root.mkdir(parents=True, exist_ok=True)

    master_keywords = load_master_keywords()
    logger.info(f"=== Group classification: {len(master_keywords)} validated keywords "
                f"(Group 1={sum(1 for k in master_keywords if k['group']==1)}, "
                f"Group 2={sum(1 for k in master_keywords if k['group']==2)}, "
                f"Group 3={sum(1 for k in master_keywords if k['group']==3)}) ===")

    work_items = []
    skipped = 0
    for p_dir in sorted(data_root.glob("year=*")):
        files = sorted(p_dir.glob("*.parquet"))
        if not files:
            continue
        year_out = out_root / p_dir.name
        year_out.mkdir(exist_ok=True)
        for pf in files:
            out_path = year_out / pf.name
            if out_path.exists():
                skipped += 1
                continue
            work_items.append((pf, out_path, config.col.content, config.col.job_id))

    if skipped:
        logger.info(f"  {skipped} file(s) already classified — skipping.")
    if not work_items:
        logger.info("  Nothing left to classify.")
        return

    n_workers = min(N_WORKERS, len(work_items))
    logger.info(f"  Classifying {len(work_items)} file(s) using {n_workers} worker process(es)...")

    done = 0
    with multiprocessing.Pool(processes=n_workers, initializer=_init_worker) as pool:
        for fname, n_rows in pool.imap_unordered(_classify_file_worker, work_items):
            done += 1
            logger.info(f"  [{done}/{len(work_items)}] {fname}: {n_rows:,} rows classified.")

    logger.info("  All files classified.")


if __name__ == "__main__":
    import argparse
    from pipeline.config_loader import load_config

    parser = argparse.ArgumentParser(description="Group-priority keyword classification")
    parser.add_argument("--config", type=str, default="pipeline/config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config(args.config)
    classify_postings_by_group(cfg)
