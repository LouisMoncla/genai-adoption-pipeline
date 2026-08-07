"""
Expand master_keywords.json with the full raw layer1/2/3 candidate pools
(keyword_lists/legacy/layer1_stanford.py, layer2_hosseini.py, layer3.py),
per explicit instruction (Louis, 2026-08-05): merge everything in now,
defer empirical (re-)validation to a later step - "let's go step by step."

IMPORTANT DISTINCTION from build_master_keywords.py: that script merges
Domenico's THESIS-VALIDATED keywords (data/raw/validated_keywords.json -
each has a real dhs_rate/avg_ratio computed against historical ad-
frequency data). This script adds the REST of the raw candidate pool that
never went through that validation - dhs_rate/avg_ratio are None, and each
new entry is marked "validated": false so they can be found/filtered/
reconsidered later without re-deriving which ones they are. Existing
entries get "validated": true added retroactively, for a consistent
schema across the whole file.

Deduplication: skip a raw candidate if its exact keyword name OR its
normalized English form already exists in master_keywords.json (catches
both literal repeats across layer1/2/3, like "ChatGPT" or "Prompt
engineering", and near-duplicates that differ only in punctuation, like
layer2's "Retrieval-augmented generation" vs layer1's already-present
"Retrieval augmented generation" - identical after normalize() strips the
hyphen).

Translations: sourced from keyword_translations.json (Google-Translate-
backed, already covers virtually the entire raw pool) via the exact same
lookup build_master_keywords.py uses - reused directly, not reimplemented.
Anything with no exact-string match is left for hand translation/flagging
rather than silently guessing.

EXPLICITLY SKIPPED, PENDING CONFIRMATION: bare "Copilot". It already went
through Domenico's validation (group 2 in validated_keywords.json) but was
deliberately EXCLUDED from master_keywords.json by Jeremias on 2026-07-30
for a documented, specific reason (aviation co-pilot collision - see
EXCLUDED_KEYWORDS in build_master_keywords.py). That's a different
situation from the rest of this script's ~80 never-reviewed additions -
reversing a specific named decision needs an explicit answer, not a
silent re-add under a generic "add everything" instruction.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "genai-adoption-pipeline" / "keyword_lists" / "master_keywords.json"
TRANSLATIONS_PATH = ROOT / "genai-adoption-pipeline" / "keyword_lists" / "translated_keywords" / "keyword_translations.json"
LEGACY_DIR = ROOT / "genai-adoption-pipeline" / "keyword_lists" / "legacy"

sys.path.insert(0, str(LEGACY_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer1_stanford import KEYWORDS_STANFORD_CORE  # noqa: E402
from layer2_hosseini import KEYWORDS_HOSSEINI_MID  # noqa: E402
from layer3 import KEYWORDS_BROAD_CANDIDATES  # noqa: E402
from build_master_keywords import _find_translation_row  # noqa: E402

sys.path.insert(0, str(ROOT / "genai-adoption-pipeline"))
from pipeline.group_classification import normalize  # noqa: E402

LANGS = ["en", "de", "fr", "it"]
LAYER_INFO = [
    (1, "St", KEYWORDS_STANFORD_CORE),
    (2, "H&L", KEYWORDS_HOSSEINI_MID),
    (3, "LLM", KEYWORDS_BROAD_CANDIDATES),
]
SKIP_NAMES = {"Copilot"}  # see module docstring - pending confirmation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write master_keywords.json (default: dry-run, report only)")
    args = parser.parse_args()

    master = json.loads(MASTER_PATH.read_text(encoding="utf-8"))["keywords"]
    translations = json.loads(TRANSLATIONS_PATH.read_text(encoding="utf-8"))

    for e in master:
        e["validated"] = True

    existing_names = {e["keyword"] for e in master}
    existing_norms = {normalize(e["forms"]["en"]) for e in master}

    new_entries = []
    skipped_dup = []
    skipped_excluded = []
    unmatched_translation = []
    seen_names = set(existing_names)
    seen_norms = set(existing_norms)

    for layer_num, source, kw_list in LAYER_INFO:
        for name in kw_list:
            if name in SKIP_NAMES:
                skipped_excluded.append((layer_num, name))
                continue
            norm = normalize(name)
            if name in seen_names or norm in seen_norms:
                skipped_dup.append((layer_num, name))
                continue
            row = _find_translation_row(translations, name)
            if row is None:
                unmatched_translation.append((layer_num, name))
                row = None  # filled in by hand afterward, not silently guessed
            new_entries.append({
                "keyword": name,
                "group": layer_num,
                "source": source,
                "dhs_rate": None,
                "avg_ratio": None,
                "validated": False,
                "forms": row,
            })
            seen_names.add(name)
            seen_norms.add(norm)

    print(f"Existing (already validated): {len(master)}")
    print(f"New candidates to add: {len(new_entries)}")
    print(f"Skipped - already present (exact name or normalized-form dup): {len(skipped_dup)}")
    for layer_num, name in skipped_dup:
        print(f"    layer{layer_num}: {name!r}")
    print(f"Skipped - explicitly excluded pending confirmation: {len(skipped_excluded)}")
    for layer_num, name in skipped_excluded:
        print(f"    layer{layer_num}: {name!r}")
    print(f"NO translation match found (needs hand translation): {len(unmatched_translation)}")
    for layer_num, name in unmatched_translation:
        print(f"    layer{layer_num}: {name!r}")

    if args.apply:
        if unmatched_translation:
            print("\nABORTING write - unmatched translations must be resolved first "
                  "(edit this script's output or fill `forms` by hand, then re-run).")
            return
        master.extend(new_entries)
        MASTER_PATH.write_text(json.dumps({"keywords": master}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {len(master)} total keywords to {MASTER_PATH}")
    else:
        print("\nDry run only - re-run with --apply to write.")


if __name__ == "__main__":
    main()
