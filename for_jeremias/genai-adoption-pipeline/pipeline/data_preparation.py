"""
Phase I: Data preparation with statistical sanity checks and column abstraction.

Processing order (correct):
  1. Filter each source file (country, recruiter, micro) → write shards
  2. Consolidate shards per year in BATCHES → deduplicate (within-batch AND
     cross-batch) → batch_NNN.parquet (memory-safe, see note below)
  3. Sanity check on the deduped batch files (column-projected reads, cheap)
  4. Remove outlier firm-years from each batch file in-place

MEMORY NOTE (added 2026-07-23): the original Step 2/4 design consolidated
each year into a single eager pl.concat(...) / pl.read_parquet(...)
DataFrame before deduping/cleaning, then wrote one "part.parquet" per year.
Tested directly against the real exported data (2023: 2.44M rows, the
largest year) on this 7.7GB-RAM machine and confirmed BOTH patterns push
system free memory toward zero and risk crashing the whole machine, not
just the Python process:
  - pl.concat([...many day-shards...]).unique(...) across a full year: system
    free memory collapsed toward ~120MB before being killed.
  - A single eager pl.read_parquet() of one already-consolidated ~2.4M-row
    file (all columns, incl. the large content_clean text + nested company/
    occupations structs): ALSO pushed free memory to ~190MB before a kill.
  - By contrast, both patterns tested safe and fast at ~60-90 day / ~400-650K
    row batch scale, and a plain multi-file pass-through concat via
    LazyFrame.sink_parquet() (no stateful op like unique()) stayed bounded
    even across a full year.
So this file never materializes a full year's rows in memory at once. Years
are processed in BATCH_DAYS-sized batches throughout, producing several
batch_NNN.parquet files per year instead of one part.parquet. Cross-batch
duplicate removal uses a lightweight ID-only running set (cheap - one column
of keys, not full rows) instead of ever re-loading prior batches' full data.
Downstream code already tolerates multiple files per year-partition
(simple_keyword_scoring.py and group_classification.py both fall back to
globbing "*.parquet" when a single "part.parquet" isn't present) -
language_detection.py needed the same fix, made alongside this one.

NOTE (2026-07-23, second occurrence): this file's fixes (this whole batching
rewrite, plus the _size_id string-cast below) were lost once already - they
existed only as uncommitted working-directory changes and got reverted by
something outside this session (git status showed the file matching the
initial commit exactly, with no later commit ever containing these changes).
Re-applied from scratch, verified against the real data again. If you're
reading this after ANOTHER unexplained revert: check `git log --oneline --
pipeline/data_preparation.py` and `git status` first - if this file doesn't
show as committed/modified with this docstring in it, the fix isn't
actually active regardless of what's described here.
"""

import logging
import math
from pathlib import Path
import gzip

import polars as pl

from pipeline.config_loader import PipelineConfig

logger = logging.getLogger(__name__)

BATCH_DAYS = 60  # ~400-450K rows/batch on this dataset - safe margin under
                  # the ~600-650K row scale confirmed safe in testing.


def prepare_data(config: PipelineConfig) -> None:
    logger.info("=== Phase I: Data Preparation ===")

    # Convert any .ndjson.gz files to .parquet first
    json_gz_files = list(config.data_path.glob("*.n*json.gz*"))
    for j_file in json_gz_files:
        p_file = j_file.parent / (j_file.name.split('.')[0] + ".parquet")
        if not p_file.exists():
            logger.info(f"Converting {j_file.name} to Parquet...")
            try:
                with gzip.open(j_file, 'rb') as f:
                    df = pl.read_ndjson(f.read(), infer_schema_length=None)
                df.write_parquet(p_file)
            except Exception as e:
                logger.error(f"Failed to convert {j_file.name}: {e}")

    parquet_files = list(config.data_path.glob("*.parquet"))
    if not parquet_files:
        logger.error(f"No Parquet files found in {config.data_path}")
        return
    logger.info(f"Found {len(parquet_files)} Parquet files")

    output_dir = Path(config.output_dir) / "prepared_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Filter each source file, write one shard per year ────────────
    logger.info("Step 1: Filtering and writing shards...")
    total_rows_written = 0
    for i, f in enumerate(parquet_files):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"  Progress: {i+1}/{len(parquet_files)} files...")

        lf_file = _flatten_and_filter(f, config)
        lf_file = lf_file.with_columns(
            pl.col(config.col.date).dt.year().alias("_partition_year")
        )
        df_file = lf_file.collect()
        if df_file.is_empty():
            continue

        for (year_val,), year_df in df_file.group_by(["_partition_year"]):
            year_path = output_dir / f"year={year_val}"
            year_path.mkdir(exist_ok=True)
            year_df.drop("_partition_year").write_parquet(year_path / f"{f.stem}.parquet")
            total_rows_written += len(year_df)

    logger.info(f"Step 1 complete: {total_rows_written:,} rows written (pre-dedup).")

    # ── Step 2: Consolidate shards → dedup → batch_NNN.parquet per year ──────
    logger.info("Step 2: Consolidating and deduplicating year partitions (in batches)...")
    total_after_dedup = 0
    for year_path in sorted(output_dir.glob("year=*")):
        source_files = sorted(
            f for f in year_path.glob("*.parquet") if not f.name.startswith("batch_")
        )
        if not source_files:
            continue

        n_rows, n_batches = _dedup_year_in_batches(year_path, source_files, config)
        total_after_dedup += n_rows

        for old_f in source_files:
            old_f.unlink()

        logger.info(f"  {year_path.name}: {n_rows:,} rows after dedup (in {n_batches} batch files)")

    logger.info(f"Step 2 complete: {total_after_dedup:,} rows after dedup.")

    # ── Step 3: Sanity check on the deduped batch files ───────────────────
    logger.info("Step 3: Running log-normal sanity check on deduped partitions...")
    outliers_df = _get_sanity_check_outliers(output_dir, config)

    # ── Step 4: Remove outlier firm-years from each batch file in-place ──────
    logger.info("Step 4: Removing outlier firms from partitions...")
    total_final = 0
    for year_path in sorted(output_dir.glob("year=*")):
        year = int(year_path.name.split("=")[1])
        outliers_year = outliers_df.filter(pl.col("_year") == year).select("_firm_id")
        year_rows = 0
        year_removed = 0
        for batch_file in sorted(year_path.glob("batch_*.parquet")):
            df = pl.read_parquet(batch_file)
            before = len(df)
            df_clean = df.join(outliers_year, on="_firm_id", how="anti")
            df_clean.write_parquet(batch_file)
            year_rows += len(df_clean)
            year_removed += before - len(df_clean)
        total_final += year_rows
        logger.info(f"  {year_path.name}: {year_rows:,} rows (removed {year_removed:,} outlier-firm rows)")

    logger.info(f"Phase I Complete: {total_final:,} rows saved to {output_dir}")


def _dedup_year_in_batches(
    year_path: Path, source_files: list[Path], config: PipelineConfig
) -> tuple[int, int]:
    """Consolidate+dedupe one year's day-shards without ever holding the
    full year in memory (see module docstring). Batches of BATCH_DAYS files
    are concatenated and deduped internally (tested safe at this scale),
    then filtered against an accumulating ID-only set to catch duplicates
    spanning batch boundaries (cheap - one column of keys, not full rows).
    Returns (total_rows, n_batch_files_written)."""
    job_id = config.col.job_id
    has_job_id = bool(job_id)
    seen_ids: set = set()
    total_rows = 0
    n_batches_written = 0

    batches = [source_files[i:i + BATCH_DAYS] for i in range(0, len(source_files), BATCH_DAYS)]
    for bi, batch_files in enumerate(batches):
        df_batch = pl.concat(
            [pl.read_parquet(f) for f in batch_files], how="diagonal_relaxed"
        )
        if has_job_id and job_id in df_batch.columns:
            df_batch = df_batch.unique(subset=[job_id], keep="first", maintain_order=False)
            df_batch = df_batch.filter(~pl.col(job_id).is_in(seen_ids))
            seen_ids.update(df_batch[job_id].to_list())

        if df_batch.is_empty():
            continue
        df_batch.write_parquet(year_path / f"batch_{bi:03d}.parquet")
        total_rows += len(df_batch)
        n_batches_written += 1

    return total_rows, n_batches_written


def _flatten_and_filter(f_path: Path, config: PipelineConfig) -> pl.LazyFrame:
    """Loads a single parquet file and applies flattening and basic filters."""
    _lf = pl.scan_parquet(f_path)
    cols_to_add = [
        pl.col(config.col.date).str.replace(" UTC$", "").str.to_datetime("%Y-%m-%d %H:%M:%S%.f").alias(config.col.date),
        pl.col(config.col.company_struct).struct.field(config.col.company_id).alias("_firm_id"),
        pl.col(config.col.company_struct).struct.field(config.col.company_name).alias("_company_name"),
        pl.col(config.col.company_struct).struct.field(config.col.company_is_recruiter).alias("_is_recruiter"),
    ]

    company_schema = _lf.collect_schema().get(config.col.company_struct)
    has_size = False
    has_metadata = False
    if company_schema is not None and getattr(company_schema, "fields", None) is not None:
        field_names = [f.name for f in company_schema.fields]
        has_size = config.col.company_size_struct in field_names
        has_metadata = config.col.company_metadata in field_names

    if has_size:
        cols_to_add.extend([
            # Cast to String: the real data has this as Int32, but
            # config.micro_enterprise_id and the sanity-check size_alpha
            # keys are strings ("57000001" etc.) - without this cast,
            # comparisons below fail with "cannot compare string with
            # numeric type" (hit during real-data testing, 2026-07-23).
            pl.col(config.col.company_struct).struct.field(config.col.company_size_struct).struct.field(config.col.company_size_id).cast(pl.String).alias("_size_id"),
            pl.col(config.col.company_struct).struct.field(config.col.company_size_struct).struct.field(config.col.company_size_name).alias("_size_name"),
        ])
    else:
        cols_to_add.extend([
            pl.lit("unknown").alias("_size_id"),
            pl.lit("unknown").alias("_size_name"),
        ])

    if has_metadata:
        # Extract the first INDUSTRY entry from company.metadata (List of structs).
        # Filter to type=="INDUSTRY", take first match's name field.
        cols_to_add.append(
            pl.col(config.col.company_struct)
            .struct.field(config.col.company_metadata)
            .list.eval(
                pl.element().filter(pl.element().struct.field("type") == "INDUSTRY")
                .struct.field("name")
            )
            .list.first()
            .alias("_industry")
        )
    else:
        cols_to_add.append(pl.lit(None).cast(pl.String).alias("_industry"))

    _lf = _lf.with_columns(cols_to_add).drop(config.col.company_struct)
    _lf = _apply_basic_filters(_lf, config)
    if config.exclude_micro_enterprises:
        _lf = _lf.filter(pl.col("_size_id") != config.micro_enterprise_id)

    return _lf


def _apply_basic_filters(lf: pl.LazyFrame, config: PipelineConfig) -> pl.LazyFrame:
    if config.country_filter:
        lf = lf.filter(
            pl.col(config.col.locations).list.eval(
                pl.element().struct.field("country") == config.country_filter
            ).list.any()
        )
    if config.date_range_start:
        lf = lf.filter(pl.col(config.col.date).dt.date() >= config.date_range_start)
    if config.date_range_end:
        lf = lf.filter(pl.col(config.col.date).dt.date() <= config.date_range_end)
    if config.exclude_recruiters:
        lf = lf.filter(pl.col("_is_recruiter") == False)
    return lf


def _get_sanity_check_outliers(prepared_dir: Path, config: PipelineConfig) -> pl.DataFrame:
    """
    Compute log-normal CI on the deduped batch files (a handful per year, not
    500 raw day-shards) — column-projected to just _firm_id/_size_id, so even
    reading every batch file across every year stays cheap (two small
    columns, not the full text/struct row data).

    Log-normal is used because firm posting counts are right-skewed
    (consistent with Gibrat's Law on firm size distributions).
    CI is computed on log(n+1) scale, bounds exponentiated back.
    """
    # Map each size_id to its alpha, then to a Z-score.
    # SME stricter (0.005), Medium moderate (0.002), Large permissive (0.001).
    # Unknown/other size classes fall back to the global sanity_ci_alpha.
    _z_map = {0.10: 1.645, 0.05: 1.96, 0.025: 2.24, 0.01: 2.576,
              0.005: 2.807, 0.002: 3.09, 0.001: 3.29}
    _fallback_alpha = round(config.sanity_ci_alpha, 3)
    _size_alpha = {
        "57000002": config.sanity_alpha_sme,
        "57000003": config.sanity_alpha_medium,
        "57000004": config.sanity_alpha_large,
    }
    # Pre-compute Z per size_id (unknown → fallback)
    _size_z = {sid: _z_map.get(round(a, 3), 1.96) for sid, a in _size_alpha.items()}
    _fallback_z = _z_map.get(_fallback_alpha, 1.96)

    all_counts = []
    for year_path in sorted(prepared_dir.glob("year=*")):
        batch_files = sorted(year_path.glob("batch_*.parquet"))
        if not batch_files:
            continue
        year = int(year_path.name.split("=")[1])
        df = pl.concat(
            [pl.read_parquet(f, columns=["_firm_id", "_size_id"]) for f in batch_files]
        )
        counts = (
            df.group_by(["_firm_id", "_size_id"])
            .agg(pl.len().alias("_n_postings"))
            .with_columns(pl.lit(year).alias("_year"))
        )
        all_counts.append(counts)

    if not all_counts:
        return pl.DataFrame({"_firm_id": [], "_year": []})

    firm_counts = pl.concat(all_counts)

    # Log-normal CI per (year, size_id) bucket — each bucket uses its own Z.
    firm_counts = firm_counts.with_columns(
        (pl.col("_n_postings") + 1).log(base=math.e).alias("_log_n"),
        # Attach per-size Z as a column so the upper-bound expression is vectorised.
        pl.col("_size_id").replace(
            old=list(_size_z.keys()),
            new=[float(z) for z in _size_z.values()],
            default=float(_fallback_z),
        ).cast(pl.Float64).alias("_z"),
    )
    stats = firm_counts.group_by(["_year", "_size_id"]).agg(
        pl.col("_log_n").mean().alias("_mean_log"),
        pl.col("_log_n").std().alias("_std_log"),
    )
    firm_counts = firm_counts.join(stats, on=["_year", "_size_id"])
    firm_counts = firm_counts.with_columns(
        # Only flag the upper tail — firms posting suspiciously many ads.
        # A firm posting very few ads is not a data quality problem.
        ((pl.col("_mean_log") + pl.col("_z") * pl.col("_std_log")).exp() - 1).alias("_upper"),
    )

    outliers = firm_counts.filter(
        pl.col("_n_postings") > pl.col("_upper")
    ).select("_firm_id", "_year")

    size_alphas_str = ", ".join(f"{sid}=a{a}" for sid, a in _size_alpha.items())
    logger.info(f"  Identified {len(outliers):,} outlier firm-year combinations "
                f"(log-normal CI per size: {size_alphas_str}).")
    return outliers
