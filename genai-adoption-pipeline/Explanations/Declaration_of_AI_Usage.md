# Declaration of AI Usage

In the interest of transparency and academic integrity, this document discloses how
AI-based tools were used in the development of the code and the accompanying documentation
in this repository.

## Tool used

**Claude Code** — Anthropic's agentic coding assistant, powered by Anthropic's Claude models
(Claude Opus) — was used during the development and documentation of the GenAI Adoption
Pipeline.

## How it was used

Claude Code was used in the following capacities:

- **Coding assistant.** It helped translate my own ideas, logic, and intended methodology into
  working code and correct syntax, and assisted with debugging and refactoring the pipeline.
- **Code commenting and review.** It was used to write and improve inline comments and
  docstrings, and to review the existing code for correctness, redundancy, and consistency —
  for example, identifying logic that had become superseded and removing it without changing
  the pipeline's results.
- **Brainstorming.** It served as a sounding board for design decisions and for weighing
  alternative approaches.
- **Documentation.** It assisted in drafting, structuring, and fact-checking the explanatory
  material in this `Explanations/` folder — in particular by reading the raw source code and
  producing detailed, line-referenced explanations of how the pipeline works.

## Documents in this folder

- **`Pipeline_Code_Explanation.md`** — drafted with Claude Code based on a direct reading of the
  pipeline source code, then reviewed and verified by me.
- **`Thesis_Overview_README.md`** — originally written by me, with Claude Code assisting in
  updating and reconciling it with the final code structure.
- **`CONFIG_TUNING_CHEATSHEET.md`** — a tuning reference for the pipeline's configuration
  parameters, prepared and refined with Claude Code's assistance.

All AI-assisted output was reviewed, verified, and — where necessary — corrected by me. The
underlying research design, methodology, and analytical decisions are my own, and I take full
responsibility for the final content of the code and documentation.

## Personal note

I can recommend using Claude Code to explain specific lines or sections of code directly in
their context. In my experience it worked well and reliably met my needs for understanding and
documenting the codebase.

— Domenico Job, 20 June 2026
