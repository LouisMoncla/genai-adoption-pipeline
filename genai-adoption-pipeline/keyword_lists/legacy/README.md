# Legacy raw keyword lists

Three hand-assembled candidate keyword lists — Stanford AI Index 2025
(`layer1_stanford.py`), Hosseini & Lichtinger 2025 (`layer2_hosseini.py`),
and an LLM-brainstormed broad list (`layer3.py`) — used only by
`pipeline/legacy/keyword_translation.py`'s Google-Translate-backed
translation cache, itself part of the superseded Phase II flow (see
`pipeline/legacy/README.md`).

**Superseded** by `keyword_lists/master_keywords.json`, built by
`code/mine/build_master_keywords.py` from Domenico's actual validated
keyword list (`data/raw/validated_keywords.json`, transcribed from his
thesis Table 4) — the currently active `pipeline/group_classification.py`
reads only `master_keywords.json`.

Kept for provenance — these are the raw candidate pools these validated
keywords were originally drawn from. Not read by anything in the active
pipeline.
