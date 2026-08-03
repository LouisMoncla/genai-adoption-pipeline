"""
GenAI Adoption Pipeline — Swiss Labor Market Analysis.

Entry point for the multi-phase pipeline:
  Phase I:  Data preparation (filtering, dedup, sanity checks)
  Language: x28 native language column (pipeline/language_detection.py — switched
            from FastText to x28's own `language` field 2026-08-02, per Jeremias)
  Translate: DeepTranslator keyword translation
  Phase II: Keyword flag + score (simplified — see below)
  Phase III: Aggregation and Reporting

Phase II note (decided with Jeremias 2026-07-17): keyword_scoring.py's
DHS-regression approach needs a 2012-2021 pre-ChatGPT baseline that this
dataset (x28, Dec 2022 - Nov 2025 only) doesn't have, so it can't run here.
main.py calls pipeline.simple_keyword_scoring.score_postings_simple instead
of pipeline.keyword_scoring.score_postings — every layer1/2/3 keyword is
treated as an equal signal (genai_flag + genai_score), no G1/G2/G3 tiering.
keyword_scoring.py itself is untouched, in case a future dataset has enough
history to bring the real regression back.
"""

import argparse
import logging
import sys
from pathlib import Path

import colorlog

import polars as pl

from pipeline.aggregation import aggregate_results
from pipeline.config_loader import load_config
from pipeline.data_preparation import prepare_data
from pipeline.simple_keyword_scoring import score_postings_simple
from pipeline.keyword_translation import translate_keywords
from pipeline.language_detection import detect_languages


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="GenAI Adoption Pipeline")
    parser.add_argument(
        "--config", type=str, default="pipeline/config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--skip-to",
        type=str,
        choices=["phase1", "lang", "translate", "phase2", "phase3"],
        help="Skip previous phases and start from this one.",
    )
    args = parser.parse_args()

    # 1. Setup Logging — colors in terminal, plain text when redirected to a file
    # Force line-by-line flushing so log files update in real time
    sys.stdout.reconfigure(line_buffering=True)
    handler = logging.StreamHandler(stream=sys.stdout)
    if sys.stdout.isatty():
        handler.setFormatter(colorlog.ColoredFormatter(
            fmt="%(asctime)s %(log_color)s[%(levelname)s]%(reset)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            },
        ))
    else:
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(handler)
    logger = logging.getLogger(__name__)

    # 2. Load Configuration
    try:
        config = load_config(args.config)
        logger.info("============================================================")
        logger.info("GenAI Adoption Pipeline — Swiss Labor Market Analysis")
        logger.info("============================================================")
        logger.info(f"Configuration loaded from {args.config}")
        _print_config_summary(config, logger)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # 3. Pipeline Execution
    try:
        # Phase I: Data Preparation
        if not args.skip_to or args.skip_to == "phase1":
            prepare_data(config)

        # Language Detection
        if not args.skip_to or args.skip_to in ["phase1", "lang"]:
            _ensure_dir(Path(config.output_dir) / "prepared_data", "Phase I")
            detect_languages(config)

        # Keyword Translation
        if not args.skip_to or args.skip_to in ["phase1", "lang", "translate"]:
            _ensure_dir(Path(config.output_dir) / "prepared_data", "Language Detection")
            
            # Use all partitioned files to get unique languages with at least 100 postings
            lf = pl.scan_parquet(Path(config.output_dir) / "prepared_data" / "year=*" / "*.parquet")
            lang_counts = lf.group_by("detected_language").agg(pl.len().alias("count")).collect()
            detected_languages = set(lang_counts.filter(pl.col("count") > 100)["detected_language"].to_list())
            
            if "unknown" in detected_languages:
                detected_languages.remove("unknown")
            translations = translate_keywords(detected_languages, config)
        else:
            # Load translations from cache if skipping
            trans_path = Path(config.output_dir) / "keyword_translations.json"
            if trans_path.exists():
                import json
                with open(trans_path, "r", encoding="utf-8") as f:
                    translations = json.load(f)["translations"]
            else:
                logger.error("Translations cache not found. Run translation phase.")
                sys.exit(1)

        # Phase II: Keyword flag + score (simplified — see module docstring above)
        if not args.skip_to or args.skip_to in ["phase1", "lang", "translate", "phase2"]:
            _ensure_dir(Path(config.output_dir) / "prepared_data", "Language Detection")
            score_postings_simple(translations, config)

        # Phase III: Aggregation
        if not args.skip_to or args.skip_to in ["phase1", "lang", "translate", "phase2", "phase3"]:
            _ensure_dir(Path(config.output_dir) / "scored_data", "Phase II")
            aggregate_results(config)

        logger.info("")
        logger.info("============================================================")
        logger.info("Pipeline Complete")
        logger.info("============================================================")

    except Exception as e:
        logger.exception(f"Pipeline failed during execution: {e}")
        sys.exit(1)


def _print_config_summary(config, logger):
    """Log a summary of the pipeline configuration."""
    logger.info("Configuration summary:")
    logger.info(f"  data_path: {config.data_path}")
    logger.info(f"  date_range: {config.date_range_start} — {config.date_range_end}")
    logger.info(f"  country_filter: {config.country_filter}")
    logger.info(f"  exclude_recruiters: {config.exclude_recruiters}")
    logger.info(f"  exclude_micro_enterprises: {config.exclude_micro_enterprises}")
    logger.info(f"  pre_period_end_year: {config.pre_period_end_year}")
    logger.info(f"  validation_year: {config.validation_year}")
    logger.info(f"  dhs_group1_threshold: {config.dhs_group1_threshold}")
    logger.info(f"  dhs_group2_threshold: {config.dhs_group2_threshold}")
    logger.info(f"  g1_occupation_coverage_threshold: {config.g1_occupation_coverage_threshold}")
    logger.info(f"  min_g1_required: {config.min_g1_required}")
    logger.info(f"  min_g2_for_rule2: {config.min_g2_for_rule2}")
    logger.info(f"  min_g2g3_for_rule3: {config.min_g2g3_for_rule3}")
    logger.info(f"  enforce_occupation_filter: {config.enforce_occupation_filter}")
    logger.info(f"  output_dir: {config.output_dir}")


def _ensure_dir(path: Path, phase_name: str):
    """Exit if a required intermediate directory is missing."""
    if not path.exists() or not any(path.iterdir()):
        logging.getLogger(__name__).error(
            f"Required directory {path} not found or empty. "
            f"Run {phase_name} first or remove --skip-to flag."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
