"""
Runs just Group Classification (Step 3) - Phase I and Language Detection are
already complete (output/prepared_data has detected_language on every row).

Must be wrapped in `if __name__ == "__main__":` because
group_classification.py now uses multiprocessing.Pool: on Windows (spawn
start method), worker processes re-import this file, and without the guard
they'd re-run the whole script recursively instead of just picking up the
worker function.
"""

import logging
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2] / "genai-adoption-pipeline")
sys.path.insert(0, ".")


def main():
    from pipeline.config_loader import load_config
    from pipeline.group_classification import classify_postings_by_group

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(handler)
    logger = logging.getLogger("run_classification_only")

    config = load_config("pipeline/config.yaml")

    logger.info("========== STEP 3: Group Classification ==========")
    classify_postings_by_group(config)
    logger.info("========== DONE ==========")


if __name__ == "__main__":
    main()
