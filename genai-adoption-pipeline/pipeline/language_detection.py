"""
Language tagging for job postings (Partitioned Data Support).

SWITCHED 2026-08-02 (per Jeremias): this used to run FastText inference over
each ad's text to guess its language. Now it just maps x28's own `language`
column (config.col.language, added to the export in
code/mine/export_duckdb_to_parquet.py) straight into `detected_language` -
x28's own tagging is more accurate than a generic langid model and free (no
model download/inference). NULL and any code not in config.yaml's
allowed_languages (including "" seen on one source dump, or clear mistags
like ja/ko/ar) fall out via the existing _restrict_to_allowed_languages()
filter below unchanged - `pl.col(...).is_in(allowed_list)` already excludes
both null and unrecognized strings, so no special-casing was needed here.
One known side effect: x28 essentially never tags "rm" (Romansh) - zero rows
on the main dump - so Romansh survivors will drop to near-zero going forward,
down from FastText's occasional (if unreliable) guesses. Worth mentioning
when reporting numbers back, not a bug.

MEMORY NOTE (kept from the FastText era, still applies): this used to
consolidate each year's Phase I output shards into a single "part.parquet"
(via eager pl.concat) before processing, then read/wrote that one file per
year. Tested directly against the real exported data (2023, the largest
year) on this 7.7GB-RAM machine: both the consolidating concat AND a plain
eager read of one already-consolidated ~2.4M-row file pushed system free
memory toward zero and risked crashing the whole machine. See
pipeline/data_preparation.py's module docstring for the full test results.
Phase I deliberately leaves each year split into several batch_NNN.parquet
files (tested safe at that scale) instead of one part.parquet - this file
processes each of those batch files individually instead of consolidating
them back into one, for the same reason. If Phase I ever produced a single
"part.parquet" (e.g. from an older run), it's still handled: this file
processes whatever *.parquet files it finds in each year directory, one at
a time.
"""

import logging
from pathlib import Path

import polars as pl

from pipeline.config_loader import PipelineConfig

logger = logging.getLogger(__name__)


def detect_languages(config: PipelineConfig) -> None:
    """
    Tag every posting's language across partitioned files, using x28's own
    `language` column (see module docstring - no inference happens here).

    Processes each *.parquet file within a year directory individually
    (batch_NNN.parquet from Phase I, or a legacy part.parquet) - never
    consolidates multiple files into one before processing. See module
    docstring for why.
    """
    logger.info("=== Language Detection (x28 native language column) ===")

    data_root = Path(config.output_dir) / "prepared_data"
    partition_dirs = sorted(data_root.glob("year=*"))

    total_processed = 0
    lang_counts = {}

    for p_dir in partition_dirs:
        year = p_dir.name.split("=")[1]
        data_files = sorted(p_dir.glob("*.parquet"))
        if not data_files:
            logger.warning(f"  Year {year}: no parquet files found, skipping.")
            continue

        year_processed = 0
        for data_file in data_files:
            # --- Skip if language tagging already done on this file ---
            schema = pl.read_parquet_schema(data_file)
            if "detected_language" in schema:
                lc = pl.read_parquet(data_file, columns=["detected_language"])
                for row in lc["detected_language"].value_counts().iter_rows():
                    lang_counts[row[0]] = lang_counts.get(row[0], 0) + row[1]
                total_processed += len(lc)
                year_processed += len(lc)
                continue

            df = pl.read_parquet(data_file).with_columns(
                pl.col(config.col.language).alias("detected_language")
            )
            df.write_parquet(data_file)

            for row in df["detected_language"].value_counts().iter_rows():
                lang_counts[row[0]] = lang_counts.get(row[0], 0) + row[1]
            total_processed += len(df)
            year_processed += len(df)

        logger.info(f"  Year {year}: {year_processed:,} postings processed ({len(data_files)} files).")

    if total_processed > 0:
        logger.info(f"Language Detection Complete. Total: {total_processed:,} postings.")
        for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {lang}: {count:,} ({count/total_processed*100:.1f}%)")
    else:
        logger.warning("Language Detection: no postings found.")

    _restrict_to_allowed_languages(config, partition_dirs)


def _restrict_to_allowed_languages(config: PipelineConfig, partition_dirs: list) -> None:
    """
    Drop postings whose detected language is not in config.allowed_languages.

    Applied right after language tagging so every downstream phase only sees the
    permitted languages (e.g. Swiss national languages + English). Idempotent:
    re-running on already-filtered partitions is a no-op. Disabled when
    allowed_languages is None/empty.
    """
    allowed = getattr(config, "allowed_languages", None)
    if not allowed:
        return

    allowed_list = list(allowed)
    logger.info(f"Restricting postings to allowed languages: {sorted(allowed_list)}")

    total_kept = 0
    total_dropped = 0
    for p_dir in partition_dirs:
        for data_file in sorted(p_dir.glob("*.parquet")):
            df = pl.read_parquet(data_file)
            before = len(df)
            df = df.filter(pl.col("detected_language").is_in(allowed_list))
            dropped = before - len(df)
            if dropped:
                df.write_parquet(data_file)
            total_kept += len(df)
            total_dropped += dropped

    logger.info(
        f"Language restriction complete: kept {total_kept:,}, "
        f"dropped {total_dropped:,} postings not in {sorted(allowed_list)}."
    )
