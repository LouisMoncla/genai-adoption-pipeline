# GenAI Adoption Pipeline

Firm-level measurement of **Generative-AI adoption in the Swiss labor market** from online
job-posting data, using econometrically validated (DHS structural-break) keyword detection.

The pipeline ingests job advertisements, identifies the postings that reference Generative-AI
tools and skills with a data-driven keyword classifier, and aggregates the results into
firm-, occupation-, industry-, and time-series-level adoption metrics.

## Start here

All documentation lives in [`Explanations/`](Explanations/):

| Document | Purpose |
|---|---|
| [Thesis_Overview_README.md](Explanations/Thesis_Overview_README.md) | High-level overview, installation, and how to adapt the pipeline to a new dataset |
| [Pipeline_Code_Explanation.md](Explanations/Pipeline_Code_Explanation.md) | Detailed, function-by-function walkthrough of the raw code |
| [CONFIG_TUNING_CHEATSHEET.md](Explanations/CONFIG_TUNING_CHEATSHEET.md) | How to make detection stricter or more lenient |
| [Declaration_of_AI_Usage.md](Explanations/Declaration_of_AI_Usage.md) | Disclosure of AI-assisted development |

## Quick start

```bash
pip install -r requirements.txt
python -m pipeline.main            # full run; configuration lives in pipeline/config.yaml
```

Run from this repository root so the `pipeline` package and the default `pipeline/config.yaml`
path resolve. Use `--skip-to {phase1,lang,translate,phase2,phase3}` to resume from a later
stage once earlier stages have completed.

## Layout

- `pipeline/` — pipeline modules and `config.yaml`
- `keyword_lists/` — source keyword definitions (three layers) and their translations
- `output/` — generated artefacts (ships with the keyword-translation cache)
- `Explanations/` — documentation (start above)

---
*Author: Domenico Job — Bachelor's thesis, University of Zurich.*
