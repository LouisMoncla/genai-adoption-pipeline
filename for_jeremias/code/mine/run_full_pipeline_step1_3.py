"""
Runs Phase I -> Language Detection -> Group Classification in sequence on the
real, combined (Dec 2020 - Nov 2025) exported dataset. See
notes/first_full_run_plan.md for the full breakdown of what each step does
and why the translate/simple-scoring phases of main.py are skipped here.
"""

import logging
import os
import sys
from pathlib import Path

# config.yaml's output_dir ("output/") and the pipeline's default config path
# ("pipeline/config.yaml") are both relative paths written assuming the
# process runs FROM WITHIN genai-adoption-pipeline/ (how pipeline/legacy/main.py,
# the original entry point before this script replaced it, was meant to be
# invoked). This script was first run from the project root instead,
# which silently created a second, wrong "output/" at the repo root and
# wrote all of Phase I's real output there - caught via `git status` showing
# an unexpected untracked output/ folder before anything got committed.
# chdir here so config.yaml's relative paths resolve the same way they
# would under that original invocation style.
os.chdir(Path(__file__).resolve().parents[2] / "genai-adoption-pipeline")
sys.path.insert(0, ".")

from pipeline.config_loader import load_config
from pipeline.data_preparation import prepare_data
from pipeline.language_detection import detect_languages
from pipeline.group_classification import classify_postings_by_group

# Guarding everything below in __main__ is required, not optional:
# classify_postings_by_group() spawns a multiprocessing.Pool internally, and
# on Windows (spawn start method) every worker process re-imports this file.
# Without this guard, each worker would re-run Steps 1-3 itself top-to-bottom
# - including spawning its OWN Pool of workers - discovered the hard way
# 2026-07-30 when a similarly-unguarded ad-hoc resume script caused ~10
# processes to concurrently re-detect languages and re-write the same
# prepared_data files, corrupting the row counts (required a full wipe and
# rerun to recover). run_classification_only.py already had this guard for
# the same reason; this file didn't because its Step 3 had never actually
# been reached in a real run before that incident (earlier runs always died
# in Steps 1-2 first).
if __name__ == "__main__":
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(handler)
    logger = logging.getLogger("run_full_pipeline")

    config = load_config("pipeline/config.yaml")

    logger.info("========== STEP 1: Phase I (data_preparation) ==========")
    prepare_data(config)

    logger.info("========== STEP 2: Language Detection ==========")
    detect_languages(config)

    logger.info("========== STEP 3: Group Classification ==========")
    classify_postings_by_group(config)

    logger.info("========== ALL STEPS COMPLETE ==========")
