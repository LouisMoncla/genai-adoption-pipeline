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
  - Apostrophes are stripped entirely (no space), so a word stays intact:
    "Domain's" -> "domains".
  - Every OTHER punctuation character (/, :, parens, HTML tags, a comma with
    no trailing space, etc.) becomes a space, same as hyphens - NOT stripped
    to nothing (fixed 2026-08-06: stripping used to glue the words on either
    side into one token, e.g. "DevOps/MLOps" -> "devopsmlops", silently
    breaking the word-boundary match regex for both halves - confirmed
    losing 33.7% of true "MLOps" occurrences this way). Only a-z, 0-9, and
    single spaces remain after this step.
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
    # BUG FIX (2026-08-06, independent audit): apostrophes and separator
    # punctuation (/, :, parens, HTML tags, comma-with-no-trailing-space,
    # ...) used to be treated identically - both stripped to nothing. That's
    # right for apostrophes (keeps a single word intact: "Domain's" ->
    # "domains", matches Jeremias's spec), but wrong for separators: deleting
    # them GLUES two distinct words together ("DevOps/MLOps" ->
    # "devopsmlops"), and the word-boundary match regex can no longer find
    # either word inside the merged token - a silent false negative, not an
    # error. Confirmed on real data: 33.7% of true "MLOps" occurrences
    # (312/927 corpus-wide) were being missed this way, purely because of a
    # neighboring "/", ":", ")", or HTML tag. Fix: strip apostrophes only
    # (unchanged from spec); replace every OTHER punctuation character with
    # a space instead of deleting it, same as hyphens already are.
    t = re.sub(r"[’']", "", t)        # apostrophes (straight + curly): strip, no space
    t = re.sub(r"[^a-z0-9 ]", " ", t)      # every other punctuation: space, not delete
    t = re.sub(r"\s+", " ", t)             # collapse any spaces introduced above
    return t.strip()


def load_master_keywords() -> list[dict]:
    data = json.loads(MASTER_KEYWORDS_PATH.read_text(encoding="utf-8"))
    return data["keywords"]


# PLURAL HANDLING (2026-08-06, per Louis): EN and FR only - deliberately NOT
# DE/IT. English adjectives never inflect for number and the head noun is
# reliably the LAST word in these short technical noun phrases ("Foundation
# Model" -> only "Model" pluralizes), so an optional trailing "s" on just the
# last word is safe and correct. French adjectives DO agree in number with
# their noun, and the head noun isn't always last - e.g. "Grand modele de
# langage" (Large language model) truly pluralizes as "Grands modeles de
# langage" (both word 1 AND word 2 take the 's'; "langage" is a "de X"
# complement and stays invariant either way) - so FR gets a per-word rule
# instead: every word gets an optional trailing "s" UNLESS it already ends
# in "s", IS itself a preposition/article, or immediately follows one
# (a prepositional complement doesn't take the head noun's plural).
# German and Italian have no single reliable suffix rule the way EN/FR do
# (German plurals depend on the noun's declension class - 7+ distinct
# patterns; Italian plurals change the final VOWEL rather than adding a
# suffix) - so instead of a blind generic rule, DE/IT get an explicit,
# hand-verified list of additional forms per keyword (2026-08-06), built by
# going through all 117 keywords individually and applying real German/
# Italian grammar rather than guessing. Reliable sub-patterns used: German
# nouns ending in -ung/-ion/-ent/-ie pluralize almost universally regular
# (-ungen/-ionen/-enten/-ien); "Modell" -> "Modelle" appears often in this
# list so is handled directly. Prepositional complements ("von X", "di X")
# stay invariant regardless of the head noun's number, same principle as
# the French "de X" handling below. Deliberately NOT covering genuine mass
# nouns (German "Lernen"/"Intelligenz"/"-ung"-as-process-noun, Italian
# "Intelligenza") - these don't pluralize in the sense used here, matching
# how English "Machine Learning" or "Intelligence" don't either. Proper
# nouns/brand names/acronyms are untouched. This is a bounded, individually-
# checked list, not an attempt at full German/Italian morphology - anything
# not listed here keeps the exact-phrase-only pattern.
_DE_EXTRA_FORMS = {
    "Vector database": ["Vektordatenbanken"],
    "Foundation Model": ["Basismodelle"],
    "Large language model": ["Große Sprachmodelle"],
    "LLM": ["Großes Sprachmodell"],  # base form is already plural; add the singular
    "Transformer-based model": ["Transformer-basierte Modelle"],
    "Diffusion Model": ["Diffusionsmodelle"],
    "Generative Model": ["Generative Modelle"],
    "Multimodal models": ["Multimodales Modell"],  # base is plural; add singular
    "Generative adversarial networks": ["Generatives gegnerisches Netzwerk"],
    "Neural Networks": ["Neuronales Netz"],
    "Virtual Assistant": ["Virtuelle Assistenten"],
    "AI Adoption": ["KI-Einführungen"],
    "Digital Transformation": ["Digitale Transformationen"],
    "AI Strategy": ["KI-Strategien"],
    "AI Alignment": ["KI-Ausrichtungen"],
    "Embeddings": ["Einbettung"],  # base is plural; add singular
    "Chain-of-Thought": ["Gedankenketten"],
}
_IT_EXTRA_FORMS = {
    "Foundation Model": ["Modelli di base"],
    "Multimodal models": ["Modello multimodale"],  # base is plural; add singular
    "Vector database": ["Banche dati vettoriali"],
    "Diffusion Model": ["Modelli di diffusione"],
    "Generative Model": ["Modelli generativi"],
    "Transformer-based model": ["Modelli basati su Transformer"],
    "Neural Networks": ["Rete neurale"],  # base is plural; add singular
    "Generative adversarial networks": ["Rete avversaria generativa"],  # base is plural; add singular
    "Virtual Assistant": ["Assistenti virtuali"],
}

_FR_INVARIANT_WORDS = {
    "de", "du", "des", "par", "pour", "avec", "en", "sur", "dans",
    "la", "le", "les", "et", "ou", "a",
}


def _plural_pattern_source(normalized_text: str, lang: str) -> str:
    """Regex source string (not yet compiled/bounded) for `normalized_text`
    that also accepts a regular plural, for lang in ("en", "fr"); exact
    literal escape, unchanged, for any other lang. See module comment above
    for the reasoning and its limits (doesn't cover irregular French
    plurals - none expected among these technical terms)."""
    words = normalized_text.split(" ")
    if lang == "en":
        if not words:
            return re.escape(normalized_text)
        parts = [re.escape(w) for w in words[:-1]]
        last = words[-1]
        parts.append(re.escape(last) if last.endswith("s") else re.escape(last) + "s?")
        return " ".join(parts)
    if lang == "fr":
        parts = []
        prev = None
        for w in words:
            if w in _FR_INVARIANT_WORDS or prev in _FR_INVARIANT_WORDS or w.endswith("s"):
                parts.append(re.escape(w))
            else:
                parts.append(re.escape(w) + "s?")
            prev = w
        return " ".join(parts)
    return re.escape(normalized_text)


_EXTRA_FORMS_BY_LANG = {"de": _DE_EXTRA_FORMS, "it": _IT_EXTRA_FORMS}


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
                (kw["keyword"], kw["group"],
                 re.compile(r"\b" + _plural_pattern_source(en_norm, "en") + r"\b"))
            )
        for lang in ("de", "fr", "it"):
            local_norm = normalize(forms.get(lang, ""))
            if local_norm and local_norm != en_norm:
                # DE/IT: alternate with any hand-verified extra forms
                # (plural/singular counterpart) for this specific keyword,
                # instead of a single exact phrase - see _DE_EXTRA_FORMS/
                # _IT_EXTRA_FORMS above.
                extra = _EXTRA_FORMS_BY_LANG.get(lang, {}).get(kw["keyword"], [])
                alts = [_plural_pattern_source(local_norm, lang)]
                alts += [re.escape(normalize(f)) for f in extra if normalize(f)]
                source = alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"
                local_patterns[lang].append(
                    (kw["keyword"], kw["group"], re.compile(r"\b" + source + r"\b"))
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
                # Row-count check, not just existence (2026-08-03): Phase I's
                # data_preparation.py has no skip-logic and rewrites every
                # batch_NNN.parquet from scratch each run - when new source
                # data shifts a year's batch boundaries (e.g. year=2018
                # tripling in size once the 2016-2018 dump was added),
                # prepared_data/year=2018/batch_000.parquet ends up with
                # totally different rows than before, but the OLD
                # classified_data/year=2018/batch_000.parquet from the prior
                # run is still sitting there under the same filename - a
                # pure existence check silently treated it as "already
                # done" forever. Caught because the post-run row count came
                # out 48,414 short of expected; the stale file's row count
                # (58,492) matched a previous run's log line exactly.
                # Cheap fix: compare row counts (parquet metadata, not a
                # full read) before trusting an existing output file.
                try:
                    src_n = pl.scan_parquet(pf).select(pl.len()).collect().item()
                    out_n = pl.scan_parquet(out_path).select(pl.len()).collect().item()
                except Exception:
                    src_n, out_n = None, None
                if src_n is not None and src_n == out_n:
                    skipped += 1
                    continue
                logger.warning(
                    f"  {p_dir.name}/{pf.name}: existing classified output "
                    f"({out_n if out_n is not None else '?'} rows) doesn't match "
                    f"current prepared_data ({src_n if src_n is not None else '?'} rows) "
                    f"- stale, reclassifying."
                )
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
