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

# config.yaml's output_dir ("output/") and main.py's own default config path
# ("pipeline/config.yaml") are both relative paths written assuming the
# process runs FROM WITHIN genai-adoption-pipeline/ (how main.py is meant to
# be invoked). This script was first run from the project root instead,
# which silently created a second, wrong "output/" at the repo root and
# wrote all of Phase I's real output there - caught via `git status` showing
# an unexpected untracked output/ folder before anything got committed.
# chdir here so config.yaml's relative paths resolve the same way they
# would under a normal `python -m pipeline.main` invocation.
os.chdir(Path(__file__).resolve().parents[2] / "genai-adoption-pipeline")
sys.path.insert(0, ".")

from pipeline.config_loader import load_config
from pipeline.data_preparation import prepare_data
from pipeline.language_detection import detect_languages
from pipeline.group_classification import classify_postings_by_group

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
