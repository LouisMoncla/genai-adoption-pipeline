"""
FastText-based language detection for job postings (Partitioned Data Support).
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

    Expects Phase I to have consolidated each year partition to a single
    part.parquet. If a year directory still contains multiple *.parquet shards
    (e.g. from a partial Phase I run), they are merged into part.parquet first
    so the rest of the pipeline always operates on one file per year.

    Performance: Uses Polars map_batches() with streaming=True to apply the
    FastText model iteratively, keeping RAM flat regardless of partition size.
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
        part_file = p_dir / "part.parquet"

        # --- Consolidate shards if Phase I left multiple files ---
        if not part_file.exists():
            shards = sorted(p_dir.glob("*.parquet"))
            if not shards:
                logger.warning(f"  Year {year}: no parquet files found, skipping.")
                continue
            if len(shards) == 1:
                shards[0].rename(part_file)
                logger.info(f"  Year {year}: renamed single shard to part.parquet.")
            else:
                logger.info(f"  Year {year}: consolidating {len(shards)} shards into part.parquet...")
                df_merged = pl.concat([pl.read_parquet(f) for f in shards], how="diagonal_relaxed")
                df_merged.write_parquet(part_file)
                for old_f in shards:
                    old_f.unlink()
                logger.info(f"  Year {year}: consolidation done ({len(df_merged):,} rows).")

        # --- Skip if language detection already done ---
        schema = pl.read_parquet_schema(part_file)
        if "detected_language" in schema:
            logger.info(f"  Year {year}: language detection already present, skipping.")
            # Still accumulate stats for the final log
            lc = pl.read_parquet(part_file, columns=["detected_language"])
            for row in lc["detected_language"].value_counts().iter_rows():
                lang_counts[row[0]] = lang_counts.get(row[0], 0) + row[1]
            total_processed += len(lc)
            continue

        logger.info(f"  Detecting languages for Year {year}...")
        lf = pl.scan_parquet(part_file)
        df = lf.with_columns(
            pl.col(config.col.content).map_batches(
                detect_lang_batch, return_dtype=pl.String
            ).alias("detected_language")
        ).collect(streaming=True)

        df.write_parquet(part_file)

        for row in df["detected_language"].value_counts().iter_rows():
            lang_counts[row[0]] = lang_counts.get(row[0], 0) + row[1]
        total_processed += len(df)
        logger.info(f"  Year {year}: {len(df):,} postings processed.")

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
        part_file = p_dir / "part.parquet"
        if not part_file.exists():
            continue
        df = pl.read_parquet(part_file)
        before = len(df)
        df = df.filter(pl.col("detected_language").is_in(allowed_list))
        dropped = before - len(df)
        if dropped:
            df.write_parquet(part_file)
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
