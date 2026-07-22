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
  - The Copilot/aviation special-case exclusion
Do not add these back in without checking first - explicitly asked for.

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
"""

import json
import logging
import re
import unicodedata
from pathlib import Path

import polars as pl

from pipeline.config_loader import PipelineConfig

logger = logging.getLogger(__name__)

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


def classify_postings_by_group(config: PipelineConfig) -> None:
    """Entry point. Reads output/prepared_data (Phase I + language detection
    output), writes output/classified_data with ad_id + group added."""
    output_dir = Path(config.output_dir)
    data_root = output_dir / "prepared_data"
    out_root = output_dir / "classified_data"
    out_root.mkdir(parents=True, exist_ok=True)

    master_keywords = load_master_keywords()
    en_patterns, local_patterns = build_keyword_patterns(master_keywords)
    logger.info(f"=== Group classification: {len(master_keywords)} validated keywords "
                f"(Group 1={sum(1 for k in master_keywords if k['group']==1)}, "
                f"Group 2={sum(1 for k in master_keywords if k['group']==2)}, "
                f"Group 3={sum(1 for k in master_keywords if k['group']==3)}) ===")

    for p_dir in sorted(data_root.glob("year=*")):
        year_name = p_dir.name
        files = sorted(p_dir.glob("*.parquet"))
        if not files:
            continue

        year_out = out_root / year_name
        year_out.mkdir(exist_ok=True)

        for pf in files:
            df = pl.read_parquet(pf)
            if df.is_empty():
                continue

            struct_series = df.select(
                pl.struct([config.col.content, "detected_language", config.col.job_id])
            ).to_series()

            ad_ids: list = []
            groups: list[str] = []
            trigger_lists: list[list[str]] = []

            for row in struct_series:
                text = row[config.col.content]
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

                ad_ids.append(row[config.col.job_id])
                trigger_lists.append(triggers)

            df_out = df.with_columns([
                pl.Series("ad_id", ad_ids),
                pl.Series("group", groups),
                pl.Series("matched_group_keywords", trigger_lists, dtype=pl.List(pl.String)),
            ])
            df_out.write_parquet(year_out / pf.name)

        logger.info(f"    {year_name}: classified.")


if __name__ == "__main__":
    import argparse
    from pipeline.config_loader import load_config

    parser = argparse.ArgumentParser(description="Group-priority keyword classification")
    parser.add_argument("--config", type=str, default="pipeline/config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config(args.config)
    classify_postings_by_group(cfg)
