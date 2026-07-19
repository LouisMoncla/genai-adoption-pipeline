"""
Phase II: Dynamic Keyword Validation and Scoring via Trend-Based Regression.

Pass structure:
  Pass 1  - Scan all prepared data; collect keyword hit counts by (year, month, lang).
             No ad-level data is kept in RAM - only aggregate counters.
  Pass 2  - Run OLS regression per keyword-language pair on the 2012-pre_period_end
             baseline; assign Group 1 / 2 / 3 / 4 via DHS + Ratio thresholds.
  Pass 2.5- Re-scan 2023+ data (post-ChatGPT period only) to compute co-occurrence
             rates and occupation-level G1 intensity. Runs after groups are known so
             no large in-memory eligible-ads dict is needed at any point.
  Pass 3  - Score every ad: apply group map + V6 flagging rules; write scored_data.
"""

import csv
import json
import logging
import math
import multiprocessing
import re
from collections import Counter
from pathlib import Path
import polars as pl
from pipeline.config_loader import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight OLS helper
# ---------------------------------------------------------------------------

class LinearRegression:
    """Simple OLS helper for trend fitting and prediction intervals."""

    def __init__(self, x: list[float], y: list[float]):
        self.n = len(x)
        if self.n < 2:
            self.beta0, self.beta1, self.se_res = 0.0, 0.0, 0.0
            self.mean_x, self.sum_sq_x = 0.0, 0.0
            return

        mean_x = sum(x) / self.n
        mean_y = sum(y) / self.n

        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den = sum((xi - mean_x) ** 2 for xi in x)

        self.beta1 = num / den if den != 0 else 0.0
        self.beta0 = mean_y - self.beta1 * mean_x

        ss_res = sum((yi - (self.beta0 + self.beta1 * xi)) ** 2 for xi, yi in zip(x, y))
        self.se_res = math.sqrt(ss_res / (self.n - 2)) if self.n > 2 else 0.0

        self.mean_x = mean_x
        self.sum_sq_x = den

    def predict(self, x_star: float) -> float:
        return self.beta0 + self.beta1 * x_star


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_COPILOT_CONTEXT_RE = re.compile(r'(?<!schemas-)\bmicrosoft\b|\bgithub\b', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_postings(translations: dict, config: PipelineConfig) -> None:
    output_dir = Path(config.output_dir)
    data_root = output_dir / 'prepared_data'

    logger.info('=== Phase II: Dynamic Keyword Validation & Scoring (Regression) ===')

    all_kws = _collect_all_keywords(translations)
    en_patterns, local_patterns = _build_origin_patterns(translations, all_kws)

    literals = _build_literal_prefilter_set(translations, all_kws)

    pass1_cache = output_dir / 'pass1_checkpoint.json'
    counts_data = None

    if pass1_cache.exists():
        try:
            counts_data = _load_pass1_cache(pass1_cache)
            logger.info('  Pass 1 checkpoint loaded (format v2).')
        except ValueError as e:
            logger.warning(f'  {e}  Deleting stale checkpoint and re-running Pass 1...')
            pass1_cache.unlink()
            counts_data = None

    if counts_data is None:
        counts_data = _collect_metrics(data_root, en_patterns, local_patterns, config,
                                       translations=translations, all_kws=all_kws,
                                       literals=literals)
        _save_pass1_cache(pass1_cache, counts_data)
        logger.info(f'  Pass 1 checkpoint saved to {pass1_cache}')

    # Check if Pass 2 + Pass 2.5 outputs already exist — skip if so
    group_map_cache = output_dir / 'keyword_group_map.json'
    g1_occ_cache = output_dir / 'g1_dominant_occupations.csv'

    if group_map_cache.exists() and g1_occ_cache.exists():
        logger.info('  Pass 2 + 2.5 checkpoints found — loading from disk, skipping to Pass 3.')
        with open(group_map_cache, 'r', encoding='utf-8') as f:
            keyword_group_map = json.load(f)
        # Load g1 intensive occupations from csv
        g1_intensive_occupations: set = set()
        with open(g1_occ_cache, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                g1_intensive_occupations.add(row['Occupation'])
        logger.info(f'  G1 intensive occupations loaded: {len(g1_intensive_occupations)}')
    else:
        # Write raw keyword counts JSON
        raw_counts_out = sorted(
            [
                {'year': k[0], 'month': k[1], 'lang': k[2], 'kw': k[3], 'count': v}
                for k, v in counts_data['counts'].items()
            ],
            key=lambda x: (x['kw'], x['year'], x['lang']),
        )
        with open(output_dir / 'raw_keyword_counts.json', 'w', encoding='utf-8') as f:
            json.dump(raw_counts_out, f)

        # Pass 2
        keyword_groups_by_lang, overall_groups, validation_report = _perform_regression_validation(
            counts_data, all_kws, config
        )

        with open(output_dir / 'validation_audit.json', 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2)

        # Pass 2.5
        final_co_occ, g1_intensive_occupations, dominance_report = _compute_cooccurrence_and_intensity(
            data_root,
            en_patterns,
            local_patterns,
            overall_groups,
            counts_data['total_occ_2025'],
            all_kws,
            config,
            literals,
        )

        keyword_group_map = _build_final_map(overall_groups, keyword_groups_by_lang, final_co_occ, all_kws)

        serializable_map = {
            k: {
                'group': v['group'],
                'group_by_lang': v['group_by_lang'],
                'g1_co_occurrence_rate': v.get('g1_co_occurrence_rate', 0.0),
                'g2_co_occurrence_rate': v.get('g2_co_occurrence_rate', 0.0),
                'g3_co_occurrence_rate': v.get('g3_co_occurrence_rate', 0.0),
                'g2_g3_partnership_rate': v.get('g2_g3_partnership_rate', 0.0),
            }
            for k, v in sorted(keyword_group_map.items())
        }

        with open(output_dir / 'keyword_group_map.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_map, f, indent=2)

        with open(output_dir / 'g1_dominant_occupations.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['Occupation', 'G1_Ads', 'Intensity', 'Cumulative_Signal_Coverage'],
            )
            writer.writeheader()
            writer.writerows(dominance_report)

    _score_all_postings(
        data_root,
        en_patterns,
        local_patterns,
        keyword_group_map,
        g1_intensive_occupations,
        config,
        literals,
        translations=translations,
        all_kws=all_kws,
    )


# ---------------------------------------------------------------------------
# Keyword collection helpers
# ---------------------------------------------------------------------------

def _build_literal_prefilter_set(translations: dict, all_kws: list[str]) -> frozenset[str]:
    """
    Build a frozenset of lowercase literal strings (min length 4) used for
    the Polars pre-filter before expensive regex matching.
    """
    lits: set = set()

    def _add(kw: str) -> None:
        if not kw or len(kw) < 4:
            return
        base = kw.lower().strip()
        lits.add(base)
        if ' ' in base:
            lits.add(base.replace(' ', '-'))
            return

    for kw in all_kws:
        _add(kw)

    for lang_data in translations.values():
        for layer in ('layer1', 'layer2', 'layer3'):
            layer_data = lang_data.get(layer, {})
            for kw in layer_data.get('keywords', []):
                _add(kw)
            for kw in layer_data.get('proper_nouns', []):
                _add(kw)

    return frozenset(l for l in lits if len(l) >= 4)


def _collect_all_keywords(translations: dict) -> list[str]:
    """Return sorted list of all unique English keyword forms."""
    all_kws: set = set()
    en_layers = translations.get('en', {})
    for layer in ('layer1', 'layer2', 'layer3'):
        all_kws.update(en_layers.get(layer, {}).get('keywords', []))
        all_kws.update(en_layers.get(layer, {}).get('proper_nouns', []))
    return sorted(all_kws)


def _build_origin_patterns(translations: dict, all_kws: list[str]) -> tuple[dict, dict]:
    """
    Build compiled regex pattern dicts keyed by canonical English keyword.

    Returns (en_patterns, local_patterns) where each maps:
        kw_en -> [compiled_pattern, ...]
    and also has a special '_combined' key:
        '_combined' -> (combined_OR_pattern, {group_name -> kw_en})
    """
    en_patterns: dict = {}
    local_patterns: dict = {lang: {} for lang in translations if lang != 'en'}

    en_form_to_kw: dict = {}
    local_form_to_kw: dict = {lang: {} for lang in local_patterns}

    for kw_en in all_kws:
        en_forms: list = []

        for ln in ('layer1', 'layer2', 'layer3'):
            en_kws = translations['en'].get(ln, {}).get('keywords', [])
            en_props = translations['en'].get(ln, {}).get('proper_nouns', [])

            if kw_en in en_kws:
                en_forms.append(en_kws[en_kws.index(kw_en)])
                continue
            if kw_en in en_props:
                en_forms.append(kw_en)

        en_forms = sorted(set(en_forms))
        en_patterns[kw_en] = [_compile_keyword_pattern(f) for f in en_forms]
        for f in en_forms:
            en_form_to_kw[f.lower()] = kw_en

        en_forms_lower = [f.lower() for f in en_forms]

        for lang in sorted(local_patterns.keys()):
            lang_forms: list = []

            for ln in ('layer1', 'layer2', 'layer3'):
                en_kws = translations['en'].get(ln, {}).get('keywords', [])
                en_props = translations['en'].get(ln, {}).get('proper_nouns', [])

                if kw_en in en_kws:
                    lang_forms.append(translations[lang][ln]['keywords'][en_kws.index(kw_en)])
                    continue
                if kw_en in en_props:
                    lang_forms.append(kw_en)

            unique_lang_forms = sorted(
                set(f for f in lang_forms if f.lower() not in en_forms_lower)
            )
            if not unique_lang_forms:
                continue

            local_patterns[lang][kw_en] = [_compile_keyword_pattern(f) for f in unique_lang_forms]
            for f in unique_lang_forms:
                local_form_to_kw[lang][f.lower()] = kw_en

    def _build_combined(form_to_kw: dict[str, str]) -> tuple:
        """Return (compiled OR-pattern, {group_name -> kw_en}).
        Sorts forms longest-first so specific patterns (e.g. 'GPT-4') take
        precedence over shorter prefixes (e.g. 'GPT') in the OR alternation.
        """
        idx_to_kw: dict = {}
        parts: list = []
        # Longest form first — prevents short patterns absorbing longer ones
        sorted_items = sorted(form_to_kw.items(), key=lambda x: -len(x[0]))
        for idx, (form, kw_en) in enumerate(sorted_items):
            gname = f'kw{idx}'
            idx_to_kw[gname] = kw_en
            inner = re.split(r'[\s\-]+', form)
            parts.append(
                '(?P<' + gname + '>\\b'
                + '[\\s\\-]+'.join(re.escape(p) for p in inner)
                + '\\b)'
            )
        combined = re.compile('|'.join(parts), re.IGNORECASE) if parts else None
        return combined, idx_to_kw

    en_combined, en_idx_to_kw = _build_combined(en_form_to_kw)
    en_patterns['_combined'] = (en_combined, en_idx_to_kw)

    for lang in local_patterns:
        lc_combined, lc_idx_to_kw = _build_combined(local_form_to_kw[lang])
        local_patterns[lang]['_combined'] = (lc_combined, lc_idx_to_kw)

    return en_patterns, local_patterns


def _compile_keyword_pattern(kw: str):
    """Compile a word-boundary regex for a single keyword form."""
    esc = re.escape(kw)
    # Replace escaped spaces/hyphens with flexible [\s\-]+ matcher
    x = '[\\s\\-]+'.join(re.split(r'[\s\-]+', esc))
    p = re.compile(f'\\b{x}\\b', re.IGNORECASE)
    return p


def _any_pattern_matches(text: str, patterns: list) -> bool:
    for p in patterns:
        if p.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# Copilot / Microsoft rule (Rule C)
# ---------------------------------------------------------------------------

def _check_copilot_microsoft_sentence(text: str, matched_kws: set[str]) -> bool:
    """
    Rule C: Flag if 'Copilot' matched AND 'Microsoft' or 'GitHub' appears anywhere
    in the job ad text (using the module-level regex).
    """
    if not text or 'Copilot' not in matched_kws:
        return False
    return bool(_COPILOT_CONTEXT_RE.search(text))


# ---------------------------------------------------------------------------
# Fast regex matching using pre-built combined OR-patterns
# ---------------------------------------------------------------------------

def _fast_match_keywords(text: str, combined_entry) -> set[str]:
    """
    Use the pre-built combined OR-pattern to find all matching canonical
    keyword keys in a single pass.
    """
    if not text:
        return set()
    combined, idx_to_kw = combined_entry
    if combined is None:
        return set()
    matched: set = set()
    for m in combined.finditer(text):
        gname = m.lastgroup
        if not gname:
            continue
        matched.add(idx_to_kw[gname])
    return matched


# ---------------------------------------------------------------------------
# Pass 1: multiprocessing year worker (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _year_scan_worker(args: tuple) -> dict:
    """
    Top-level worker for multiprocessing. Receives plain serializable args,
    rebuilds regex patterns inside the worker (can't pickle compiled regex),
    scans all parquet files for one year, returns aggregated counter dicts.

    Optimisation: uses Polars vectorised ops for global/totals/occ counts,
    and a literal pre-filter to skip ~99% of rows before the Python regex loop.

    translations_arg may be either:
      - a dict  (legacy / small datasets)
      - a str   (path to a temp JSON file; loaded here to avoid pickling large dict)
    """
    import os
    os.environ['POLARS_MAX_THREADS'] = '1'  # prevent per-worker thread explosion

    year_dir_str, translations_arg, all_kws, literals_list, date_col, content_col, validation_year = args

    # Load translations from disk if a path string was passed — avoids pickling
    # the large dict to every worker simultaneously (main RAM spike cause)
    if isinstance(translations_arg, str):
        import json as _json
        with open(translations_arg, 'r', encoding='utf-8') as _fh:
            translations = _json.load(_fh)
    else:
        translations = translations_arg
    from pathlib import Path as _Path

    year_dir = _Path(year_dir_str)
    part_file = year_dir / 'part.parquet'
    files = [part_file] if part_file.exists() else sorted(year_dir.glob('*.parquet'))
    if not files:
        return {'counts': {}, 'totals': {}, 'global': {}, 'occ_2025': {}}

    # Rebuild patterns inside worker (regex objects can't be pickled)
    en_patterns, local_patterns = _build_origin_patterns(translations, all_kws)

    # Build Polars literal pre-filter expressions
    lit_exprs = [
        pl.col('_cl').str.contains(lit, literal=True)
        for lit in literals_list
    ]

    counts_map: dict = {}
    totals_map: dict = {}
    global_totals: dict = {}
    occ_2025: dict = {}

    for pf in files:
        df = pl.read_parquet(pf)
        if len(df) == 0:
            continue

        # Add year/month columns once via Polars (fast)
        df = df.with_columns([
            pl.col(date_col).dt.year().cast(pl.Int32).alias('_year'),
            pl.col(date_col).dt.month().cast(pl.Int32).alias('_month'),
        ]).filter(pl.col(date_col).is_not_null() & pl.col(content_col).is_not_null())

        if df.is_empty():
            continue

        # --- Global totals (all rows, fast Polars group_by) ---
        for row in df.group_by(['_year', '_month']).agg(pl.len().alias('n')).iter_rows(named=True):
            global_totals[(row['_year'], row['_month'])] = \
                global_totals.get((row['_year'], row['_month']), 0) + row['n']

        # --- Language totals (all rows) ---
        for row in df.group_by(['_year', '_month', 'detected_language']).agg(pl.len().alias('n')).iter_rows(named=True):
            totals_map[(row['_year'], row['_month'], row['detected_language'])] = \
                totals_map.get((row['_year'], row['_month'], row['detected_language']), 0) + row['n']

        # --- Occupation counts for validation year (all rows, Polars) ---
        val_df = df.filter(pl.col('_year') == validation_year)
        if not val_df.is_empty():
            for row in val_df.iter_rows(named=True):
                occs = row['occupations']
                ad_occ = occs[0].get('name') if occs and len(occs) > 0 else None
                if ad_occ:
                    occ_2025[ad_occ] = occ_2025.get(ad_occ, 0) + 1

        # --- Literal pre-filter: skip rows with no keyword literals ---
        df_cl = df.with_columns(
            pl.col(content_col).str.to_lowercase().fill_null('').alias('_cl')
        )
        pre_mask = df_cl.select(
            pl.any_horizontal(*lit_exprs).fill_null(False).alias('_any')
        )['_any']
        df_pre = df.filter(pre_mask)

        if df_pre.is_empty():
            continue

        # --- Python regex loop only on pre-filtered rows (~1-5% of total) ---
        struct_series = df_pre.select(
            pl.struct(['_year', '_month', content_col, 'detected_language'])
        ).to_series()

        for row in struct_series:
            year = row['_year']
            month = row['_month']
            text = row[content_col]
            ad_lang = row['detected_language']

            for kw_en in _fast_match_keywords(text, en_patterns['_combined']):
                k = (year, month, 'en', kw_en)
                counts_map[k] = counts_map.get(k, 0) + 1

            if ad_lang != 'en' and ad_lang in local_patterns and '_combined' in local_patterns[ad_lang]:
                for kw_en in _fast_match_keywords(text, local_patterns[ad_lang]['_combined']):
                    k = (year, month, ad_lang, kw_en)
                    counts_map[k] = counts_map.get(k, 0) + 1

    return {'counts': counts_map, 'totals': totals_map, 'global': global_totals, 'occ_2025': occ_2025}


# ---------------------------------------------------------------------------
# Pass 1: scan worker and aggregator
# ---------------------------------------------------------------------------

def _metrics_scan_worker(
    s: pl.Series,
    en_patterns: dict,
    local_patterns: dict,
    config: PipelineConfig,
) -> dict:
    """
    Iterate over a struct Series (one row = one job ad) and accumulate
    keyword hit counts.  Returns a dict with keys:
        counts   - {(year, month, lang, kw): int}
        totals   - {(year, month, lang): int}
        global   - {(year, month): int}
        occ_2025 - {occupation_name: int}  (validation_year only)
    """
    batch_counts: dict = {}
    batch_totals: dict = {}
    batch_global: dict = {}
    batch_occ_2025: dict = {}

    for row in s:
        dt = row[config.col.date]
        text = row[config.col.content]
        ad_lang = row['detected_language']
        occs = row['occupations']

        if not dt or not text:
            continue

        year = dt.year
        month = dt.month

        # Global counter (year, month)
        batch_global[(year, month)] = batch_global.get((year, month), 0) + 1
        # Language-level total
        batch_totals[(year, month, ad_lang)] = batch_totals.get((year, month, ad_lang), 0) + 1

        # English keyword matches
        for kw_en in _fast_match_keywords(text, en_patterns['_combined']):
            k = (year, month, 'en', kw_en)
            batch_counts[k] = batch_counts.get(k, 0) + 1

        # Local-language keyword matches
        if ad_lang != 'en' and ad_lang in local_patterns and '_combined' in local_patterns[ad_lang]:
            for kw_en in _fast_match_keywords(text, local_patterns[ad_lang]['_combined']):
                k = (year, month, ad_lang, kw_en)
                batch_counts[k] = batch_counts.get(k, 0) + 1

        # Occupation counts (validation year only)
        if year == config.validation_year:
            if occs and len(occs) > 0:
                ad_occ = occs[0].get('name')
            else:
                ad_occ = None
            if not ad_occ:
                continue
            batch_occ_2025[ad_occ] = batch_occ_2025.get(ad_occ, 0) + 1

    return {
        'counts': batch_counts,
        'totals': batch_totals,
        'global': batch_global,
        'occ_2025': batch_occ_2025,
    }


def _collect_metrics(
    data_root: Path,
    en_patterns: dict,
    local_patterns: dict,
    config: PipelineConfig,
    translations: dict = None,
    all_kws: list = None,
    literals: frozenset = None,
) -> dict:
    logger.info('  Pass 1: Collecting counts (parallel year-by-year)...')

    year_dirs = sorted(data_root.glob('year=*'))

    # Use multiprocessing if translations/all_kws available (patterns rebuilt in workers)
    use_mp = translations is not None and all_kws is not None
    n_workers = min(len(year_dirs), 3)

    counts_map: dict = {}
    totals_map: dict = {}
    global_totals: dict = {}
    total_occ_2025: dict = {}

    if use_mp and n_workers > 1:
        logger.info(f'  Pass 1: Using {n_workers} parallel workers for {len(year_dirs)} years...')
        literals_list = sorted(literals)

        # Save translations to a temp file so workers load from disk instead of
        # receiving a pickled copy each — eliminates the N×large-dict RAM spike
        import tempfile, os
        tmp_fd, tmp_translations_path = tempfile.mkstemp(suffix='.json', prefix='translations_')
        try:
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as _tf:
                json.dump(translations, _tf)

            args_list = [
                (str(p_dir), tmp_translations_path, all_kws, literals_list,
                 config.col.date, config.col.content, config.validation_year)
                for p_dir in year_dirs
            ]
            with multiprocessing.Pool(processes=n_workers) as pool:
                results = pool.map(_year_scan_worker, args_list)
        finally:
            try:
                os.unlink(tmp_translations_path)
            except OSError:
                pass

        for p_dir, part in zip(year_dirs, results):
            for k, v in part['counts'].items():
                counts_map[k] = counts_map.get(k, 0) + v
            for k, v in part['totals'].items():
                totals_map[k] = totals_map.get(k, 0) + v
            for k, v in part['global'].items():
                global_totals[k] = global_totals.get(k, 0) + v
            for k, v in part['occ_2025'].items():
                total_occ_2025[k] = total_occ_2025.get(k, 0) + v
            logger.info(f'    {p_dir.name}: scanned.')
    else:
        # Fallback: sequential (no translations passed or single core)
        for p_dir in year_dirs:
            part_file = p_dir / 'part.parquet'
            files = [part_file] if part_file.exists() else sorted(p_dir.glob('*.parquet'))
            if not files:
                continue

            for pf in files:
                df = pl.read_parquet(pf)
                if len(df) == 0:
                    continue

                struct_series = df.select(
                    pl.struct([
                        config.col.date,
                        config.col.content,
                        'detected_language',
                        'occupations',
                    ])
                ).to_series()

                part = _metrics_scan_worker(struct_series, en_patterns, local_patterns, config)

                for k, v in part['counts'].items():
                    counts_map[k] = counts_map.get(k, 0) + v
                for k, v in part['totals'].items():
                    totals_map[k] = totals_map.get(k, 0) + v
                for k, v in part['global'].items():
                    global_totals[k] = global_totals.get(k, 0) + v
                for k, v in part['occ_2025'].items():
                    total_occ_2025[k] = total_occ_2025.get(k, 0) + v

            logger.info(f'    {p_dir.name}: scanned.')

    return {
        'counts': counts_map,
        'totals': totals_map,
        'global': global_totals,
        'total_occ_2025': total_occ_2025,
    }


# ---------------------------------------------------------------------------
# Pass 2: OLS regression & group assignment
# ---------------------------------------------------------------------------

def _perform_regression_validation(
    metrics: dict,
    all_kws: list[str],
    config: PipelineConfig,
) -> tuple[dict, dict, list]:
    """
    Assign every keyword-language pair to Group 1, 2, or 3 via DHS + Ratio.
    Returns (keyword_groups_by_lang, overall_groups, validation_report).
    """
    logger.info('  Pass 2: Running econometric validation (Lin-Lin OLS + DHS)...')

    counts = metrics['counts']

    pre_years = range(2012, config.pre_period_end_year + 1)
    val_year = config.validation_year

    overall_groups: dict = {}
    keyword_groups_by_lang: dict = {}
    validation_report: list = []

    langs: list = list(('en', 'de', 'fr', 'it', 'rm', 'sq', 'es', 'pt'))

    for kw_en in sorted(all_kws):
        lang_groups: dict = {}

        for lang in langs:
            x_pre: list = []
            y_pre: list = []
            pre_period_dict: dict = {}

            for y in pre_years:
                y_hits = sum(
                    counts.get((y, m, lang, kw_en), 0)
                    for m in range(1, 13)
                )
                x_pre.append(float(y - 2012))
                y_pre.append(float(y_hits))
                pre_period_dict[y] = y_hits

            if len(x_pre) < 2:
                lang_groups[lang] = 3
                continue

            # Validation year hits (all months)
            y_hits_2025 = float(sum(
                counts.get((val_year, m, lang, kw_en), 0)
                for m in range(1, 13)
            ))

            # Baseline average 2012-2018
            sum_12_18 = sum(pre_period_dict[y] for y in range(2012, 2019))
            avg_12_18 = sum_12_18 / 7.0

            div_by_zero = avg_12_18 == 0
            ratio = 0.0 if div_by_zero else y_hits_2025 / avg_12_18

            # OLS regression on pre-period
            reg = LinearRegression(x_pre, y_pre)
            y_pred = reg.predict(float(val_year - 2012))

            denom_dhs = y_hits_2025 + max(0.0, y_pred)
            dhs = (
                2.0 * (y_hits_2025 - y_pred) / denom_dhs
                if denom_dhs > 0
                else 0.0
            )

            # Total hits across all languages in validation year
            total_hits_2025_all_langs = float(sum(
                counts.get((val_year, m, l, kw_en), 0)
                for l in langs
                for m in range(1, 13)
            ))

            # Group assignment
            # G1: zero pre-period baseline OR DHS >= 1.95
            # G2: DHS >= 1.72 AND (zero baseline OR ratio > 11.75)
            # G3: DHS >= 1.00 AND (zero baseline OR ratio > 20)  — helper keywords
            # G4: everything else — recorded but contributes nothing to flagging rules
            if div_by_zero and dhs >= config.dhs_group1_threshold:
                assigned_group = 1
            elif dhs >= config.dhs_group2_threshold and (div_by_zero or ratio > config.ratio_group2_threshold):
                assigned_group = 2
            elif dhs >= config.dhs_group3_threshold and (div_by_zero or ratio > config.ratio_group3_threshold):
                assigned_group = 3
            else:
                assigned_group = 4

            # Minimum signal check: demote G1/G2 with too few 2025 hits
            if assigned_group in (1, 2) and total_hits_2025_all_langs < 3:
                assigned_group = 3

            lang_groups[lang] = assigned_group

            # Build full year-by-year hit dict
            all_year_hits = dict(pre_period_dict)
            for y in range(config.pre_period_end_year + 1, val_year + 1):
                all_year_hits[y] = sum(
                    counts.get((y, m, lang, kw_en), 0)
                    for m in range(1, 13)
                )

            validation_report.append({
                'kw': kw_en,
                'lang': lang,
                'hits_12_21': sum(all_year_hits.values()),
                'hits_12_18': sum_12_18,
                'avg_hits_12_18': avg_12_18,
                'average_ratio': ratio,
                'beta0': reg.beta0,
                'beta1': reg.beta1,
                'y_pred_2025': y_pred,
                'y_actual_2025': y_hits_2025,
                'dhs': dhs,
                'assigned_group': assigned_group,
                'yearly_hits': {str(y): int(v) for y, v in sorted(all_year_hits.items())},
            })

        # English-dominant rule:
        #   G4 in English → force ALL languages to G4 (no signal confirmed in English)
        #   G3 in English → cap ALL languages at G3 (only a weak helper)
        #   G1/G2 in English → each language keeps its own DHS-assigned group
        en_group = lang_groups.get('en', 4)
        if en_group == 4:
            final_groups_by_lang = {l: 4 for l in langs}
        elif en_group == 3:
            final_groups_by_lang = {l: 3 for l in langs}
        else:
            final_groups_by_lang = dict(lang_groups)

        keyword_groups_by_lang[kw_en] = final_groups_by_lang
        overall_groups[kw_en] = en_group

    return keyword_groups_by_lang, overall_groups, validation_report


# ---------------------------------------------------------------------------
# Pass 2.5: co-occurrence and occupation intensity
# ---------------------------------------------------------------------------

def _compute_cooccurrence_and_intensity(
    data_root: Path,
    en_patterns: dict,
    local_patterns: dict,
    overall_groups: dict,
    total_occ_2025: dict,
    all_kws: list[str],
    config: PipelineConfig,
    literals: frozenset[str],
) -> tuple[dict, set, list]:
    logger.info('  Pass 2.5: Computing co-occurrence and occupation intensity (2023+ scan)...')

    content_col = config.col.content

    g1_kws = {kw for kw, g in overall_groups.items() if g == 1}
    g2_kws = {kw for kw, g in overall_groups.items() if g == 2}
    g3_kws = {kw for kw, g in overall_groups.items() if g == 3}

    g1_list = sorted(g1_kws)
    g2_list = sorted(g2_kws)
    g3_list = sorted(g3_kws)

    lit_list = sorted(literals)
    lit_exprs = [
        pl.col('_cl').str.contains(lit, literal=True)
        for lit in lit_list
    ]

    co_occ_metrics: dict = {
        kw: {'co_occ_g1': 0, 'co_occ_g2': 0, 'co_occ_g3': 0, 'co_occ_g2_or_g3': 0, 'total': 0}
        for kw in all_kws
    }

    # NOTE: the '_2025' suffix in this and related names (occ_2025, total_occ_2025)
    # is historical — these counters actually track config.validation_year, which is
    # configurable, not literally 2025. The string key 'total_occ_2025' is also a
    # persisted pass1-checkpoint field, so it is kept as-is to avoid invalidating
    # existing checkpoints.
    g1_occ_2025_hits = Counter()

    for p_dir in sorted(data_root.glob('year=*')):
        year = int(p_dir.name.split('=')[1])
        if year < 2023:
            continue

        part_file = p_dir / 'part.parquet'
        if part_file.exists():
            files = [part_file]
        else:
            files = sorted(p_dir.glob('*.parquet'))

        if not files:
            continue

        for pf in files:
            df = pl.read_parquet(pf, columns=[content_col, 'detected_language', 'occupations'])

            if df.is_empty():
                continue

            # Build pre-filter mask
            pre_mask = (
                df.select(
                    pl.col(content_col)
                    .str.to_lowercase()
                    .fill_null('')
                    .alias('_cl')
                )
                .with_columns(
                    pl.any_horizontal(*lit_exprs).fill_null(False).alias('_any')
                )['_any']
            )

            df_pre = df.filter(pre_mask)
            if df_pre.is_empty():
                continue

            texts = df_pre[content_col].to_list()
            langs = df_pre['detected_language'].to_list()

            matched_kws_list: list = []
            for text, lang in zip(texts, langs):
                if not text:
                    matched_kws_list.append([])
                    continue

                m = _fast_match_keywords(text, en_patterns['_combined'])

                lang = lang or 'en'
                if lang != 'en' and lang in local_patterns and '_combined' in local_patterns[lang]:
                    m |= _fast_match_keywords(text, local_patterns[lang]['_combined'])

                matched_kws_list.append(sorted(m))

            df_pre = df_pre.with_columns(
                pl.Series(
                    '_matched_kws',
                    matched_kws_list,
                    dtype=pl.List(pl.String),
                )
            )

            df_m = df_pre.filter(
                pl.col('_matched_kws').list.len() > 0
            )

            if df_m.is_empty():
                continue

            # Compute co-occurrence flags per keyword per ad
            df_exp = (
                df_m.with_columns([
                    pl.col('_matched_kws').list.eval(
                        pl.element().is_in(g1_list)
                    ).list.sum().cast(pl.Int32).alias('_n_g1'),
                    pl.col('_matched_kws').list.eval(
                        pl.element().is_in(g2_list)
                    ).list.sum().cast(pl.Int32).alias('_n_g2'),
                    pl.col('_matched_kws').list.eval(
                        pl.element().is_in(g3_list)
                    ).list.sum().cast(pl.Int32).alias('_n_g3'),
                ])
                .select(list(('_matched_kws', '_n_g1', '_n_g2', '_n_g3')))
                .explode('_matched_kws')
                .rename({'_matched_kws': '_kw'})
                .with_columns([
                    (pl.col('_kw').is_in(g1_list).cast(pl.Int32)).alias('_kw_g1'),
                    (pl.col('_kw').is_in(g2_list).cast(pl.Int32)).alias('_kw_g2'),
                    (pl.col('_kw').is_in(g3_list).cast(pl.Int32)).alias('_kw_g3'),
                ])
                .with_columns([
                    (
                        ((pl.col('_n_g1') > 0) & (pl.col('_kw_g1') == 0))
                        | (pl.col('_n_g1') > 1)
                    ).alias('_co_g1'),
                    (
                        ((pl.col('_n_g2') > 0) & (pl.col('_kw_g2') == 0))
                        | (pl.col('_n_g2') > 1)
                    ).alias('_co_g2'),
                    (
                        ((pl.col('_n_g3') > 0) & (pl.col('_kw_g3') == 0))
                        | (pl.col('_n_g3') > 1)
                    ).alias('_co_g3'),
                ])
                .with_columns(
                    (pl.col('_co_g2') | pl.col('_co_g3')).alias('_co_g2_or_g3')
                )
            )

            agg = (
                df_exp.group_by('_kw')
                .agg([
                    pl.len().alias('total'),
                    pl.col('_co_g1').sum().alias('co_occ_g1'),
                    pl.col('_co_g2').sum().alias('co_occ_g2'),
                    pl.col('_co_g3').sum().alias('co_occ_g3'),
                    pl.col('_co_g2_or_g3').sum().alias('co_occ_g2_or_g3'),
                ])
            )

            for row in agg.iter_rows(named=True):
                kw = row['_kw']
                if kw not in co_occ_metrics:
                    continue
                s = co_occ_metrics[kw]
                s['total'] += row['total']
                s['co_occ_g1'] += row['co_occ_g1']
                s['co_occ_g2'] += row['co_occ_g2']
                s['co_occ_g3'] += row['co_occ_g3']
                s['co_occ_g2_or_g3'] += row['co_occ_g2_or_g3']

            # Occupation G1 intensity (validation year only)
            if year == config.validation_year:
                g1_ads = df_m.with_columns(
                    pl.col('_matched_kws').list.eval(
                        pl.element().is_in(g1_list)
                    ).list.sum().cast(pl.Int32).alias('_n_g1')
                ).filter(pl.col('_n_g1') > 0)
                if not g1_ads.is_empty():
                    occ_counts = (
                        g1_ads.select(
                            pl.col('occupations')
                            .list.first()
                            .struct.field('name')
                            .alias('_occ')
                        )
                        .filter(
                            pl.col('_occ').is_not_null() & (pl.col('_occ') != '')
                        )
                        .group_by('_occ')
                        .agg(pl.len().alias('cnt'))
                    )

                    for row in occ_counts.iter_rows(named=True):
                        g1_occ_2025_hits[row['_occ']] += row['cnt']

        logger.info(f'    {p_dir.name}: co-occurrence scan done.')

    # Compute final co-occurrence rates
    final_co_occ: dict = {}
    for kw, s in co_occ_metrics.items():
        total = s['total']
        final_co_occ[kw] = {
            'g1_rate': s['co_occ_g1'] / total if total > 0 else 0.0,
            'g2_rate': s['co_occ_g2'] / total if total > 0 else 0.0,
            'g3_rate': s['co_occ_g3'] / total if total > 0 else 0.0,
            'g2_g3_partnership_rate': s['co_occ_g2_or_g3'] / total if total > 0 else 0.0,
        }

    # Build intensity stats list
    intensity_stats: list = []
    for occ, total_ads in total_occ_2025.items():
        g1_hits = g1_occ_2025_hits.get(occ, 0)
        intensity = g1_hits / total_ads if total_ads > 0 else 0.0
        intensity_stats.append({
            'Occupation': occ,
            'G1_Hits_2025': g1_hits,
            'Total_Ads_2025': total_ads,
            'Intensity': intensity,
        })

    intensity_stats.sort(key=lambda x: x['Intensity'], reverse=True)

    with open(Path(config.output_dir) / 'occupation_intensity_stats.json', 'w', encoding='utf-8') as f:
        json.dump(intensity_stats, f, indent=2)

    # Identify g1_intensive_occupations using coverage accumulation
    g1_intensive_occupations: set = set()
    total_signal = sum(g1_occ_2025_hits.values())
    target_signal = total_signal * config.g1_occupation_coverage_threshold

    current_signal = 0
    dominance_report: list = []

    for s in intensity_stats:
        if s['Intensity'] < config.min_occupation_intensity:
            break
        g1_hits = s['G1_Hits_2025']
        g1_intensive_occupations.add(s['Occupation'])
        current_signal += g1_hits
        dominance_report.append({
            'Occupation': s['Occupation'],
            'G1_Ads': g1_hits,
            'Intensity': f"{s['Intensity']:.2%}",
            'Cumulative_Signal_Coverage': (
                f'{current_signal / total_signal:.2%}'
                if total_signal > 0
                else '0%'
            ),
        })
        if current_signal >= target_signal:
            break

    covered = current_signal / total_signal if total_signal > 0 else 0.0
    logger.info(
        f'    Identified {len(g1_intensive_occupations)} intensive occupations covering '
        f'{covered:.1%} of G1 signal (floor >={config.min_occupation_intensity:.0%}'
        f', target {config.g1_occupation_coverage_threshold:.0%}).'
    )

    return final_co_occ, g1_intensive_occupations, dominance_report


# ---------------------------------------------------------------------------
# Pass 2 finaliser: merge everything into keyword_group_map
# ---------------------------------------------------------------------------

def _build_final_map(
    overall_groups: dict,
    keyword_groups_by_lang: dict,
    final_co_occ: dict,
    all_kws: list[str],
) -> dict:
    final_map: dict = {}
    for kw in sorted(all_kws):
        m = final_co_occ.get(kw, {})
        final_map[kw] = {
            'group': overall_groups[kw],
            'group_by_lang': keyword_groups_by_lang[kw],
            'g1_co_occurrence_rate': m.get('g1_rate', 0.0),
            'g2_co_occurrence_rate': m.get('g2_rate', 0.0),
            'g3_co_occurrence_rate': m.get('g3_rate', 0.0),
            'g2_g3_partnership_rate': m.get('g2_g3_partnership_rate', 0.0),
        }
    return final_map


# ---------------------------------------------------------------------------
# Pass 3: score all postings
# ---------------------------------------------------------------------------

def _score_all_postings(
    data_root: Path,
    en_patterns: dict,
    local_patterns: dict,
    group_map: dict,
    g1_intensive_occupations: set,
    config: PipelineConfig,
    literals: frozenset[str],
    translations: dict = None,
    all_kws: list = None,
    output_dir=None,
) -> None:
    logger.info('  Pass 3: Scoring all partitions...')
    output_dir = Path(config.output_dir)
    scored_root = output_dir / 'scored_data'
    scored_root.mkdir(parents=True, exist_ok=True)

    year_dirs = sorted(data_root.glob('year=*'))
    use_mp = translations is not None and all_kws is not None
    n_workers = min(len(year_dirs), 3)

    if use_mp and n_workers > 1:
        import tempfile, os as _os
        # Save large dicts to temp files — avoid pickling them to every worker
        tmp_fd_t, tmp_trans = tempfile.mkstemp(suffix='.json', prefix='trans_p3_')
        tmp_fd_g, tmp_gmap = tempfile.mkstemp(suffix='.json', prefix='gmap_p3_')
        try:
            with _os.fdopen(tmp_fd_t, 'w', encoding='utf-8') as _f:
                json.dump(translations, _f)
            with _os.fdopen(tmp_fd_g, 'w', encoding='utf-8') as _f:
                json.dump(group_map, _f)

            g1_list = sorted(g1_intensive_occupations)
            literals_list = sorted(literals)

            args_list = [
                (str(p_dir), tmp_trans, all_kws,
                 tmp_gmap,
                 g1_list, literals_list,
                 config.col.content, 'detected_language', 'occupations',
                 config.min_g1_required, config.min_g2_for_rule2, config.min_g2g3_for_rule3,
                 config.enforce_occupation_filter,
                 str(scored_root))
                for p_dir in year_dirs
            ]

            logger.info(f'  Pass 3: Using {n_workers} parallel workers for {len(year_dirs)} years...')
            with multiprocessing.Pool(processes=n_workers) as pool:
                for year_name in pool.imap_unordered(_year_score_worker, args_list):
                    logger.info(f'    {year_name.split("=")[1]}: scored.')
        finally:
            for _p in (tmp_trans, tmp_gmap):
                try:
                    _os.unlink(_p)
                except OSError:
                    pass
    else:
        # Sequential fallback
        lit_list = sorted(literals)
        lit_exprs = [
            pl.col('_cl').str.contains(lit, literal=True)
            for lit in lit_list
        ]

        for p_dir in year_dirs:
            year = int(p_dir.name.split('=')[1])

            part_file = p_dir / 'part.parquet'
            files = [part_file] if part_file.exists() else sorted(p_dir.glob('*.parquet'))
            year_out = scored_root / f'year={year}'
            year_out.mkdir(exist_ok=True)

            for pf in files:
                if not pf.exists():
                    continue
                df = pl.read_parquet(pf)
                if len(df) == 0:
                    continue

                pre_mask = (
                    df.select(
                        pl.col(config.col.content)
                        .str.to_lowercase()
                        .fill_null('')
                        .alias('_cl')
                    )
                    .with_columns(
                        pl.any_horizontal(*lit_exprs).fill_null(False).alias('_any')
                    )['_any']
                )
                pre_match_set: set[int] = set(pre_mask.arg_true().to_list())

                struct_series = df.select(
                    pl.struct([config.col.content, 'detected_language', 'occupations'])
                ).to_series()

                scored_series = _score_batch_worker(
                    struct_series, en_patterns, local_patterns,
                    group_map, g1_intensive_occupations, config,
                    pre_match_set=pre_match_set,
                )

                df_scored = df.with_columns(
                    scored_series.alias('_results')
                ).unnest('_results')

                for _col in ('matched_keywords', 'matched_g1_keywords',
                             'matched_g2_keywords', 'matched_g3_keywords'):
                    if _col in df_scored.columns:
                        df_scored = df_scored.with_columns(
                            pl.col(_col).cast(pl.List(pl.String))
                        )

                df_scored.write_parquet(year_out / pf.name)

            logger.info(f'    {year}: scored.')


def _score_batch_worker(
    s: pl.Series,
    en_patterns: dict,
    local_patterns: dict,
    group_map: dict,
    g1_intensive_occupations: set,
    config: PipelineConfig,
    pre_match_set: set[int] | None = None,
) -> pl.Series:
    """
    Score each ad with the V6 Triple-Lock flagging rules:
      Rule 1: >= min_g1_required Group 1 keywords  (self-sufficient flag)
      Rule 2: >= min_g2_for_rule2 G2 keywords, AND — if
              enforce_occupation_filter is True — the occupation is AI-intensive
      Rule 3: >= min_g2g3_for_rule3 combined G2+G3 keywords AND at least 1 G2
      Rule C: Copilot + Microsoft/GitHub sentence context
    """
    _default = {
        'genai_flag': False,
        'flag_rule_1': False,
        'flag_rule_2': False,
        'flag_rule_3': False,
        'flag_rule_copilot': False,
        'group1_match': False,
        'group2_match': False,
        'g1_count': 0,
        'g2_count': 0,
        'partner_count': 0,
        'matched_keywords': [],
        'matched_g1_keywords': [],
        'matched_g2_keywords': [],
        'matched_g3_keywords': [],
    }

    res: list = []

    for i, row in enumerate(s):
        # Skip rows that didn't pass the literal pre-filter
        if pre_match_set is not None and i not in pre_match_set:
            res.append(_default)
            continue

        text = row[config.col.content]
        ad_lang = row['detected_language']
        occs = row['occupations']

        matches: list = []

        # English matches
        for kw_en in _fast_match_keywords(text, en_patterns['_combined']):
            matches.append((kw_en, 'en'))

        # Local language matches
        if (
            ad_lang != 'en'
            and ad_lang in local_patterns
            and '_combined' in local_patterns[ad_lang]
        ):
            for kw_en in _fast_match_keywords(text, local_patterns[ad_lang]['_combined']):
                matches.append((kw_en, ad_lang))

        m_g1: set = set()
        m_g2: set = set()
        m_g3: set = set()
        all_matched_kws: set = set()

        for kw, m_lang in matches:
            g = group_map.get(kw, {}).get('group_by_lang', {}).get(m_lang, 3)
            all_matched_kws.add(kw)
            if g == 1:
                m_g1.add(kw)
            elif g == 2:
                m_g2.add(kw)
            elif g == 3:
                # G3 == the helper set by definition (every group-3 keyword), so
                # membership in group 3 is sufficient — no extra helper check needed.
                m_g3.add(kw)

        # First occupation name
        ad_occ = None
        if occs and len(occs) > 0:
            ad_occ = occs[0].get('name')

        # Rule 1
        flag_r1 = len(m_g1) >= config.min_g1_required

        # Aviation exclusion for Rule 2
        _AVIATION_OCCS = frozenset({'Pilot/in', 'Kopilot/in', 'Co-Pilot/in'})
        if ad_occ in _AVIATION_OCCS:
            m_g2_for_r2 = {k for k in m_g2 if k.lower() != 'copilot'}
        else:
            m_g2_for_r2 = m_g2

        partner_count = len(m_g2)

        # Rule 2 (occupation gate optional via config.enforce_occupation_filter)
        flag_r2 = (
            len(m_g2_for_r2) >= config.min_g2_for_rule2
            and (not config.enforce_occupation_filter or ad_occ in g1_intensive_occupations)
        )

        # Rule 3
        flag_r3 = (len(m_g2) + len(m_g3)) >= config.min_g2g3_for_rule3 and len(m_g2) >= 1

        # Rule C (Copilot + Microsoft/GitHub)
        flag_rc = _check_copilot_microsoft_sentence(text, all_matched_kws)

        flag = flag_r1 or flag_r2 or flag_r3 or flag_rc

        res.append({
            'genai_flag': flag,
            'flag_rule_1': flag_r1,
            'flag_rule_2': flag_r2,
            'flag_rule_3': flag_r3,
            'flag_rule_copilot': flag_rc,
            'group1_match': len(m_g1) > 0,
            'group2_match': len(m_g2) > 0,
            'g1_count': len(m_g1),
            'g2_count': len(m_g2),
            'partner_count': partner_count,
            'matched_keywords': sorted(all_matched_kws),
            'matched_g1_keywords': sorted(m_g1),
            'matched_g2_keywords': sorted(m_g2),
            'matched_g3_keywords': sorted(m_g3),
        })

    return pl.Series(res)


# ---------------------------------------------------------------------------
# Pass 3: top-level year-scoring worker (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _year_score_worker(args: tuple) -> str:
    """
    Top-level worker for parallel Pass 3. Receives serializable args only.
    Rebuilds regex patterns inside the worker; loads group_map + translations
    from temp JSON files to avoid large-dict pickling overhead.

    This is the parallel twin of _score_batch_worker (the sequential fallback).
    The flagging logic below (Rules 1/2/3 + Copilot Rule C, aviation exclusion,
    group bucketing) is duplicated by necessity and MUST be kept in sync with
    _score_batch_worker if the scoring rules change.

    Returns the year string (e.g. '2019') for logging.
    """
    import os
    os.environ['POLARS_MAX_THREADS'] = '1'

    (year_dir_str, translations_path, all_kws,
     group_map_path,
     g1_intensive_list, literals_list,
     content_col, lang_col, occ_col,
     min_g1, min_g2_r2, min_g2_r3,
     enforce_occ,
     scored_root_str) = args

    import json as _json
    from pathlib import Path as _Path

    with open(translations_path, 'r', encoding='utf-8') as _f:
        translations = _json.load(_f)
    with open(group_map_path, 'r', encoding='utf-8') as _f:
        group_map = _json.load(_f)

    g1_intensive: set = set(g1_intensive_list)

    en_patterns, local_patterns = _build_origin_patterns(translations, all_kws)

    year_dir = _Path(year_dir_str)
    year = int(year_dir.name.split('=')[1])
    scored_root = _Path(scored_root_str)
    year_out = scored_root / f'year={year}'
    year_out.mkdir(parents=True, exist_ok=True)

    part_file = year_dir / 'part.parquet'
    files = [part_file] if part_file.exists() else sorted(year_dir.glob('*.parquet'))

    lit_exprs = [
        pl.col('_cl').str.contains(lit, literal=True)
        for lit in literals_list
    ]

    _COPILOT_RE = re.compile(r'(?<!schemas-)\bmicrosoft\b|\bgithub\b', re.IGNORECASE)
    _AVIATION_OCCS = frozenset({'Pilot/in', 'Kopilot/in', 'Co-Pilot/in'})

    _default = {
        'genai_flag': False, 'flag_rule_1': False, 'flag_rule_2': False,
        'flag_rule_3': False, 'flag_rule_copilot': False,
        'group1_match': False, 'group2_match': False,
        'g1_count': 0, 'g2_count': 0, 'partner_count': 0,
        'matched_keywords': [], 'matched_g1_keywords': [],
        'matched_g2_keywords': [], 'matched_g3_keywords': [],
    }

    for pf in files:
        if not pf.exists():
            continue
        df = pl.read_parquet(pf)
        if len(df) == 0:
            continue

        pre_mask = (
            df.select(
                pl.col(content_col).str.to_lowercase().fill_null('').alias('_cl')
            )
            .with_columns(pl.any_horizontal(*lit_exprs).fill_null(False).alias('_any'))
            ['_any']
        )
        pre_match_set: set = set(pre_mask.arg_true().to_list())

        struct_series = df.select(
            pl.struct([content_col, lang_col, occ_col])
        ).to_series()

        res: list = []
        for i, row in enumerate(struct_series):
            if i not in pre_match_set:
                res.append(_default)
                continue

            text = row[content_col]
            ad_lang = row[lang_col]
            occs = row[occ_col]

            matches: list = []
            for kw_en in _fast_match_keywords(text, en_patterns['_combined']):
                matches.append((kw_en, 'en'))
            if (ad_lang != 'en' and ad_lang in local_patterns
                    and '_combined' in local_patterns[ad_lang]):
                for kw_en in _fast_match_keywords(text, local_patterns[ad_lang]['_combined']):
                    matches.append((kw_en, ad_lang))

            m_g1: set = set()
            m_g2: set = set()
            m_g3: set = set()
            all_kws_matched: set = set()
            for kw, m_lang in matches:
                g = group_map.get(kw, {}).get('group_by_lang', {}).get(m_lang, 3)
                all_kws_matched.add(kw)
                if g == 1:
                    m_g1.add(kw)
                elif g == 2:
                    m_g2.add(kw)
                elif g == 3:
                    # G3 == the helper set by definition; group membership suffices.
                    m_g3.add(kw)

            ad_occ = occs[0].get('name') if occs and len(occs) > 0 else None

            flag_r1 = len(m_g1) >= min_g1
            m_g2_r2 = {k for k in m_g2 if k.lower() != 'copilot'} if ad_occ in _AVIATION_OCCS else m_g2
            flag_r2 = len(m_g2_r2) >= min_g2_r2 and (not enforce_occ or ad_occ in g1_intensive)
            flag_r3 = (len(m_g2) + len(m_g3)) >= min_g2_r3 and len(m_g2) >= 1
            flag_rc = bool(text and 'Copilot' in all_kws_matched and _COPILOT_RE.search(text))

            res.append({
                'genai_flag': flag_r1 or flag_r2 or flag_r3 or flag_rc,
                'flag_rule_1': flag_r1, 'flag_rule_2': flag_r2,
                'flag_rule_3': flag_r3, 'flag_rule_copilot': flag_rc,
                'group1_match': len(m_g1) > 0, 'group2_match': len(m_g2) > 0,
                'g1_count': len(m_g1), 'g2_count': len(m_g2),
                'partner_count': len(m_g2),
                'matched_keywords': sorted(all_kws_matched),
                'matched_g1_keywords': sorted(m_g1),
                'matched_g2_keywords': sorted(m_g2),
                'matched_g3_keywords': sorted(m_g3),
            })

        df_scored = df.with_columns(
            pl.Series(res).alias('_results')
        ).unnest('_results')

        for _col in ('matched_keywords', 'matched_g1_keywords',
                     'matched_g2_keywords', 'matched_g3_keywords'):
            if _col in df_scored.columns:
                df_scored = df_scored.with_columns(
                    pl.col(_col).cast(pl.List(pl.String))
                )

        df_scored.write_parquet(year_out / pf.name)

    return year_dir.name


# ---------------------------------------------------------------------------
# Pass 1 checkpoint serialisation / deserialisation
# ---------------------------------------------------------------------------

def _save_pass1_cache(path: Path, counts_data: dict) -> None:
    """Serialise Pass 1 aggregate counts to disk (tuple keys -> pipe-separated strings)."""
    cache = {
        'format_version': 2,
        'counts': {
            f'{k[0]}|{k[1]}|{k[2]}|{k[3]}': v
            for k, v in counts_data['counts'].items()
        },
        'totals': {
            f'{k[0]}|{k[1]}|{k[2]}': v
            for k, v in counts_data['totals'].items()
        },
        'global': {
            f'{k[0]}|{k[1]}': v
            for k, v in counts_data['global'].items()
        },
        'total_occ_2025': counts_data.get('total_occ_2025', {}),
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f)


def _load_pass1_cache(path: Path) -> dict:
    """Reload Pass 1 checkpoint, reconstructing tuple keys.  Raises ValueError on stale format."""
    with open(path, 'r', encoding='utf-8') as f:
        cache = json.load(f)

    if cache.get('format_version', 1) < 2:
        raise ValueError('Stale pass1 checkpoint (v1 format \u2014 pre-refactor).')

    counts = {
        (int(p[0]), int(p[1]), p[2], p[3]): v
        for key, v in cache['counts'].items()
        for p in [key.split('|', 3)]
    }

    totals = {
        (int(p[0]), int(p[1]), p[2]): v
        for key, v in cache['totals'].items()
        for p in [key.split('|', 2)]
    }

    global_totals = {
        (int(p[0]), int(p[1])): v
        for key, v in cache['global'].items()
        for p in [key.split('|', 1)]
    }

    return {
        'counts': counts,
        'totals': totals,
        'global': global_totals,
        'total_occ_2025': cache.get('total_occ_2025', {}),
    }
