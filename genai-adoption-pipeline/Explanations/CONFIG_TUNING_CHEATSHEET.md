# Config Tuning Cheat-Sheet — Making AI Detection Stricter or More Lenient

All knobs live in `config.yaml`. This sheet explains **which parameters actually
change the classification**, in which direction, and which parameters in the file
are inert (loaded but never used) so you don't waste time tuning them.

The algorithm has **two stages**, and you can tune either one:

1. **Group assignment** (Phase II, per keyword) — decides whether each keyword
   becomes a Group 1 / 2 / 3 signal or is discarded (Group 4).
2. **Ad flagging rules** (Phase II, per ad) — combine the matched keywords in an
   ad into the final `genai_flag` (AI vs. non-AI).

---

## 1. How an ad gets flagged (the logic you are tuning)

A keyword is assigned a group from its post-ChatGPT growth:

| Group | Condition (English form drives the keyword's overall group) |
|-------|-------------------------------------------------------------|
| **G1** | zero pre-period baseline **AND** `DHS ≥ dhs_group1_threshold` |
| **G2** | `DHS ≥ dhs_group2_threshold` **AND** (zero baseline **OR** `ratio > ratio_group2_threshold`) |
| **G3** (helper) | `DHS ≥ dhs_group3_threshold` **AND** (zero baseline **OR** `ratio > ratio_group3_threshold`) |
| **G4** | everything else → never contributes to a flag |

(`DHS` = Davis–Haltiwanger–Schuh growth rate, range −2…+2. `ratio` = 2025 hits ÷
mean 2012–2018 hits.) A G1/G2 keyword with **< 3 total hits in the validation
year** is demoted to G3 — see the hardcoded caveat at the bottom.

An ad is flagged `genai_flag = True` if **any** rule fires:

| Rule | Fires when | Controlling parameter |
|------|-----------|-----------------------|
| **Rule 1** | ad contains ≥ `min_g1_required` Group-1 keywords | `min_g1_required` |
| **Rule 2** | ad contains ≥ `min_g2_for_rule2` Group-2 keywords **AND** the ad's occupation is in the AI-intensive set | `min_g2_for_rule2`, plus `min_occupation_intensity` / `g1_occupation_coverage_threshold` |
| **Rule 3** | (#G2 + #G3) ≥ `min_g2g3_for_rule3` **AND** at least one G2 | `min_g2g3_for_rule3` |
| **Rule C** | "Copilot" matched **AND** "Microsoft"/"GitHub" anywhere in the text | *(no config knob — edit code)* |

---

## 2. To make detection STRICTER (flag fewer ads as AI)

Most effective first:

| Change | Effect |
|--------|--------|
| `min_g2g3_for_rule3: 3 → 4` (or higher) | Rule 3 needs more co-occurring keywords; cuts the most false positives. |
| `dhs_group2_threshold: 1.72 → higher` (e.g. 1.85) | Fewer keywords qualify as G2. |
| `ratio_group2_threshold: 11.8 → higher` | Same — demands a sharper post-ChatGPT jump. |
| `dhs_group3_threshold` ↑ and `ratio_group3_threshold` ↑ | Shrinks the G3 helper pool, weakening Rule 3. |
| `min_occupation_intensity: 0.02 → higher` (e.g. 0.05) | Fewer occupations count as "AI-intensive", so Rule 2 fires less. |
| `g1_occupation_coverage_threshold: 0.8 → lower` (e.g. 0.6) | Smaller AI-intensive occupation set → Rule 2 stricter. |
| `min_g2_for_rule2: 1 → 2` | Rule 2 needs two G2 keywords, not one. |
| `enforce_occupation_filter: true` (default) | Keeps Rule 2's occupation gate on — the stricter setting. |
| `dhs_group1_threshold: 1.95 → higher` | Stricter G1 (note: G1 also **requires** a zero pre-2022 baseline, so this knob has limited reach). |

**Strictest practical setup:** raise `min_g2g3_for_rule3` to 4, set
`min_g2_for_rule2` to 2, raise `min_occupation_intensity` to ~0.05, keep
`enforce_occupation_filter: true`. This leans the measure heavily on Rule 1
(unambiguous G1 terms like ChatGPT).

## 3. To make detection MORE LENIENT (flag more ads as AI)

| Change | Effect |
|--------|--------|
| `enforce_occupation_filter: true → false` | Rule 2 fires on the G2 keyword count alone, ignoring occupation. **Strong loosener** — with `min_g2_for_rule2: 1` this means *any single G2 keyword* flags the ad. |
| `min_g2g3_for_rule3: 3 → 2` | Two co-occurring G2/G3 keywords are enough. Big single loosener. |
| `dhs_group2_threshold` / `ratio_group2_threshold` ↓ | More keywords promoted to G2. |
| `dhs_group3_threshold` / `ratio_group3_threshold` ↓ | Larger G3 helper pool. |
| `min_occupation_intensity: 0.02 → lower` (e.g. 0.01) | More occupations are "AI-intensive" → Rule 2 fires more. |
| `g1_occupation_coverage_threshold: 0.8 → higher` (e.g. 0.9) | Larger AI-intensive occupation set. |

---

## 4. Time windows — set these to match the data, not strictness

| Parameter | Meaning | Watch out |
|-----------|---------|-----------|
| `pre_period_end_year` | Last "clean" pre-ChatGPT year for the OLS baseline (default 2021). | Keep ≤ 2021 so late-2022 noise stays out of the baseline. |
| `validation_year` | Year used to measure the shock (default 2025). | **Must be a fully-populated year in the ETH x28 snapshot.** If their data ends mid-2025 or in 2024, set this to the last complete year, or every keyword's growth will look artificially small. |

---

## 5. Cleaned-up parameters (no longer present)

These parameters existed in earlier versions but did nothing, and have been
**removed** from `config.yaml` / the loader so they no longer mislead:
`min_g2_required`, `min_g3_required`, `ratio_group1_threshold`,
`regression_alpha`, `pre_period_growth_limit`, `helper_co_occurrence_threshold`.
(The real Rule-2/Rule-3 knobs are `min_g2_for_rule2` and `min_g2g3_for_rule3`.)
`enforce_occupation_filter`, previously inert, is now actually wired into Rule 2 —
see §2/§3.

## 6. ⚠️ Behaviour fixed in code, not in config

If you need to change these, edit `pipeline/keyword_scoring.py`:

- **Minimum-signal floor:** a G1/G2 keyword with fewer than **3** validation-year hits (summed over all languages) is demoted to G3. The `3` is hardcoded in `_perform_regression_validation`.
- **English-dominant rule:** if a keyword is G4 in English, it is forced to G4 in *all* languages; if G3 in English, it is capped at G3 everywhere. Only G1/G2-in-English keywords keep per-language groups.
- **Rule C (Copilot):** the "Copilot + Microsoft/GitHub" logic and the aviation-pilot exclusion list (`Pilot/in`, `Kopilot/in`, `Co-Pilot/in`) are hardcoded.
- The keyword lists themselves live in `keyword_lists/layer1_stanford.py`, `layer2_hosseini.py`, `layer3.py`. Editing them changes detection more than any threshold — but note the translation cache is keyed by a hash of these three files, so **editing a keyword list invalidates `output/keyword_translations.json` and triggers re-translation** (needs internet) unless you also refresh the cache.
