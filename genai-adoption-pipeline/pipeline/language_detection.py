"""
FastText-based language detection for job postings (Partitioned Data Support).

MEMORY NOTE (added 2026-07-23): this used to consolidate each year's Phase I
output shards into a single "part.parquet" (via eager pl.concat) before
processing, then read/wrote that one file per year. Tested directly against
the real exported data (2023, the largest year) on this 7.7GB-RAM machine:
both the consolidating concat AND a plain eager read of one already-
consolidated ~2.4M-row file pushed system free memory toward zero and risked
crashing the whole machine. See pipeline/data_preparation.py's module
docstring for the full test results. Phase I now deliberately leaves each
year split into several batch_NNN.parquet files (tested safe at that scale)
instead of one part.parquet — this file processes each of those batch files
individually instead of consolidating them back into one, for the same
reason. If Phase I ever produced a single "part.parquet" (e.g. from an older
run), it's still handled: this file processes whatever *.parquet files it
finds in each year directory, one at a time.

NOTE (2026-07-23, second occurrence): this fix was lost once already (existed
only as an uncommitted change, reverted by something outside this session -
see pipeline/data_preparation.py's docstring for the fuller story) and had to
be re-applied from scratch. If you're reading this after another unexplained
revert, check `git log --oneline -- pipeline/language_detection.py` first.
"""

import logging
from pathlib import Path

import fasttext
import polars as pl

from pipeline.config_loader import PipelineConfig

logger = logging.getLogger(__name__)


def detect_languages(config: PipelineConfig) -> None:
    """
    Detect language for all job ads across partitioned files.

    Processes each *.parquet file within a year directory individually
    (batch_NNN.parquet from Phase I, or a legacy part.parquet) - never
    consolidates multiple files into one before processing. See module
    docstring for why.

    Performance: Uses Polars map_batches() with streaming=True to apply the
    FastText model iteratively within each file.
    """
    logger.info("=== Language Detection ===")

    model_path = Path(config.output_dir) / ".cache" / "lid.176.ftz"
    if not model_path.exists():
        _download_model(model_path)

    model = fasttext.load_model(str(model_path))
    logger.info("FastText model loaded")

    data_root = Path(config.output_dir) / "prepared_data"
    partition_dirs = sorted(data_root.glob("year=*"))

    total_processed = 0
    lang_counts = {}

    def detect_lang_batch(s: pl.Series) -> pl.Series:
        texts = [str(t).replace("\n", " ") if t is not None else "" for t in s]
        if not texts:
            return pl.Series(dtype=pl.String)
        predictions = model.predict(texts, k=1)
        return pl.Series([p[0].replace("__label__", "") for p in predictions[0]])

    for p_dir in partition_dirs:
        year = p_dir.name.split("=")[1]
        data_files = sorted(p_dir.glob("*.parquet"))
        if not data_files:
            logger.warning(f"  Year {year}: no parquet files found, skipping.")
            continue

        year_processed = 0
        for data_file in data_files:
            # --- Skip if language detection already done on this file ---
            schema = pl.read_parquet_schema(data_file)
            if "detected_language" in schema:
                lc = pl.read_parquet(data_file, columns=["detected_language"])
                for row in lc["detected_language"].value_counts().iter_rows():
                    lang_counts[row[0]] = lang_counts.get(row[0], 0) + row[1]
                total_processed += len(lc)
                year_processed += len(lc)
                continue

            lf = pl.scan_parquet(data_file)
            df = lf.with_columns(
                pl.col(config.col.content).map_batches(
                    detect_lang_batch, return_dtype=pl.String
                ).alias("detected_language")
            ).collect(streaming=True)

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

    Applied right after language detection so every downstream phase only sees the
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


def _download_model(dest: Path) -> None:
    """Download FastText lid.176.ftz if missing."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    logger.info(f"Downloading FastText model from {url}...")
    urllib.request.urlretrieve(url, dest)
