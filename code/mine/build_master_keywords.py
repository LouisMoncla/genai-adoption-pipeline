"""
Merge Domenico's validated keyword-group assignments (data/raw/validated_keywords.json,
transcribed from thesis Table 4) with the existing per-language translated keyword forms
(genai-adoption-pipeline/keyword_lists/translated_keywords/keyword_translations.json) into
one master file: keyword_lists/master_keywords.json.

Matching approach: translations.json stores English keyword text as the canonical key
(layer1/2/3 -> keywords/proper_nouns, same list length and order across all 4 languages).
Every validated keyword was checked by hand against the English lists and matches by exact
string - so this script matches by exact (case-sensitive) string equality against the English
"keywords" and "proper_nouns" arrays across layer1/2/3, then pulls the same-index de/fr/it
form. Proper nouns (brand names like ChatGPT, LLM, Copilot) are stored identically across all
4 languages in translations.json already - i.e. "not translated" - so no separate fallback
logic is needed for them; using the same-index lookup handles it automatically.

If a validated keyword has no match anywhere in translations.json, its EN form is used for
every language and a warning is printed - this should not happen given the check above, but
the script doesn't assume it silently.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATED_PATH = ROOT / "data" / "raw" / "validated_keywords.json"
TRANSLATIONS_PATH = ROOT / "genai-adoption-pipeline" / "keyword_lists" / "translated_keywords" / "keyword_translations.json"
OUT_PATH = ROOT / "genai-adoption-pipeline" / "keyword_lists" / "master_keywords.json"

LANGS = ["en", "de", "fr", "it"]
LAYERS = ["layer1", "layer2", "layer3"]


def _find_translation_row(translations: dict, keyword: str) -> dict | None:
    """Find (layer, list_type, index) of `keyword` in the EN lists, return that
    row's form in every language. None if not found anywhere."""
    en = translations["translations"]["en"]
    for layer in LAYERS:
        for list_type in ("keywords", "proper_nouns"):
            en_list = en.get(layer, {}).get(list_type, [])
            if keyword in en_list:
                idx = en_list.index(keyword)
                row = {}
                for lang in LANGS:
                    lang_list = translations["translations"][lang].get(layer, {}).get(list_type, [])
                    row[lang] = lang_list[idx] if idx < len(lang_list) else keyword
                return row
    return None


def main():
    validated = json.loads(VALIDATED_PATH.read_text(encoding="utf-8"))
    translations = json.loads(TRANSLATIONS_PATH.read_text(encoding="utf-8"))

    master = []
    unmatched = []

    for entry in validated["keywords"]:
        kw = entry["keyword"]
        row = _find_translation_row(translations, kw)
        if row is None:
            unmatched.append(kw)
            row = {lang: kw for lang in LANGS}

        master.append({
            "keyword": kw,
            "group": entry["group"],
            "source": entry["source"],
            "dhs_rate": entry["dhs_rate"],
            "avg_ratio": entry["avg_ratio"],
            "forms": row,  # {"en": ..., "de": ..., "fr": ..., "it": ...}
        })

    if unmatched:
        print(f"WARNING: {len(unmatched)} validated keyword(s) had no match in "
              f"keyword_translations.json, English form used for all languages: {unmatched}")
    else:
        print(f"All {len(master)} validated keywords matched a translation entry.")

    by_group = {}
    for e in master:
        by_group.setdefault(e["group"], 0)
        by_group[e["group"]] += 1
    print(f"Group counts: {by_group}")

    OUT_PATH.write_text(json.dumps({"keywords": master}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
