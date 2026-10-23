# AI-Use Log

Required by PLAN §8 and feeds the AI-use disclosure appendix of the report. Add one row for every AI-assisted piece of work, the same day. "Verified" says what a person checked and how; "Not verified" is an allowed answer, but the work cannot be merged until someone has verified it.

| Date | Member | Tool | Purpose | Output (files, commit) | Verified by and how |
|---|---|---|---|---|---|
| 2026-10-15 | A | Claude Code (Claude Opus) | Wrote the v1 code, tests and config | `src/`, `tests/`, `config/`, commit `3bb33f4` | Tests pass; **not yet read line by line by the owners** (W1-W4) |
| 2026-10-15 | A | Claude Code (Claude Opus) | Drafted the plan and the literature review | `docs/PLAN.md`, `docs/RESEARCH.md` | Some references are still marked [unverified]; team review due W1 |
| 2026-10-19 | A | Claude Code (Claude Opus) | Made the template narrator the default, guarded A4 behind `--with-llm`, added STATUS.md | commit `b7a355c` | Tests `test_default_narrator_is_the_free_template`, `test_a4_is_not_run_without_with_llm_even_when_a_key_is_set` |
| 2026-10-23 | A | Claude Code (Claude Opus) | Rewrote STATUS.md as an evidence checklist; wrote README, CI workflow, this log | `docs/STATUS.md`, `README.md`, `.github/workflows/ci.yml`, `docs/AI_USE_LOG.md` | Full test suite passed; README install steps run on a clean copy; evaluation rerun to back the STATUS numbers |

## Rules

- Never paste credentials, API keys or personal data into an AI tool.
- The report is written by the team, not by an AI (PLAN §9, academic integrity). AI may be used for checking and suggestions, and each use is logged here.
- Every reference an AI suggests is checked at its source before it goes into RESEARCH.md or the report.
