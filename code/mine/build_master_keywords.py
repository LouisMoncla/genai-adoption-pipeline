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

OVERRIDES (added 2026-07-23, Jeremias's decision): the bare acronyms "LLM" and "RAG" were
found to collide with unrelated Swiss professional terms after normalization strips
punctuation - "LL.M." (the Master of Laws legal degree) becomes indistinguishable from "LLM"
(84% of all "LLM" matches in the first real run turned out to be the legal degree, not the AI
term), and "RAG" collided with a Swiss auditor licensing designation on a smaller scale. Fix:
match on the spelled-out forms instead of the bare acronyms for these two specific validated
keywords. The "keyword" label (used for traceability to Domenico's Table 4, and shown in
matched_group_keywords output) stays "LLM"/"RAG" - only the "forms" (the actual text matched
against) change. Applied as a post-merge override here, not by editing
keyword_lists/layer2_hosseini.py - that file isn't in this system's actual matching path
(group_classification.py reads master_keywords.json, built from validated_keywords.json +
keyword_translations.json; layer1/2/3.py and keyword_translation.py's Google-Translate-backed
cache are only used by the older keyword_scoring.py / simple_keyword_scoring.py path). Editing
this file's OVERRIDES dict and re-running is the only step needed - no retranslation, no
internet required.
Note: "RAG" -> "Retrieval-Augmented Generation" is Group 1, same as the already-existing
"Retrieval augmented generation" (St source) validated keyword - this override makes "RAG"
match the identical phrase, which is harmless redundancy (same group either way) rather than
a functional change. "LLM" -> "Large language models" is the consequential one: it's Group 2,
while "Large language model" (singular) is a separate Group 1 keyword - kept deliberately
distinct (plural form here) so this override doesn't make the Group 2 entry unreachable by
duplicating the Group 1 singular form.
Per Jeremias: check the match count after this change - if the spelled-out forms give very
few matches (people overwhelmingly write the bare acronym in practice), that's a real finding
to report back, not a sign the fix needs more work.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATED_PATH = ROOT / "data" / "raw" / "validated_keywords.json"
TRANSLATIONS_PATH = ROOT / "genai-adoption-pipeline" / "keyword_lists" / "translated_keywords" / "keyword_translations.json"
OUT_PATH = ROOT / "genai-adoption-pipeline" / "keyword_lists" / "master_keywords.json"

LANGS = ["en", "de", "fr", "it"]
LAYERS = ["layer1", "layer2", "layer3"]

# Manually translated (not from keyword_translations.json - these exact plural/expanded forms
# aren't in that file). German/Italian pluralization is regular here; French uses the
# standard "grands modèles de langage" phrasing. Worth a native-speaker spot-check but these
# are straightforward technical terms, not idiomatic.
FORM_OVERRIDES = {
    "LLM": {
        "en": "Large language models",
        "de": "Große Sprachmodelle",
        "fr": "Grands modèles de langage",
        "it": "Modelli linguistici di grandi dimensioni",
    },
    "RAG": {
        "en": "Retrieval-Augmented Generation",
        "de": "Abrufgestützte Generierung",
        "fr": "Génération augmentée par récupération",
        "it": "Generazione aumentata dal recupero",
    },
}


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

        if kw in FORM_OVERRIDES:
            row = dict(FORM_OVERRIDES[kw])

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
