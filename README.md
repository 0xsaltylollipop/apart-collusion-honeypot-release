# Honey, I Shrunk the Eval: A Short Collusion Trap Predicts Which LLMs Collude in the Long Run

Code, labels and results for a project in the [Apart Research AI Collusion Research Sprint](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25) (Track 3.2). Paper: see the Apart sprint submission. All markets are simulated.

**The question.** AI developers use honeypots, tests that tempt a model to misbehave, as safety evidence. Does a model's behaviour in a short collusion honeypot predict its behaviour in long games? In the honeypot, a scripted rival (firm C) offers two copies of a model to fix prices; in long games (self-play), three copies compete for 300 rounds. Nine LLMs, ranked by how often and how early they agree to fix prices.

## Results

- **The short test predicts the long games:** Spearman ρ = 0.93 (95% interval 0.58–1.00, one-sided p < 0.001, n = 9; agreements judged by Claude Opus 5.5, checked against human labels).
- **The offer matters:** it ranks models more closely than the chat before the offer (ρ = 0.62), because it draws out models that rarely speak unprompted (exploratory; nine models cannot yet confirm the gap).
- **Prices are mixed:** opening-round profit gains track self-play profits (ρ = 0.72, descriptive); the planned offer-minus-no-offer price measure does not (ρ = −0.23, exploratory after its planted-colluder check failed).

Full output: [`results/RESULTS.md`](results/RESULTS.md) · Figure 2: [`results/figure2.pdf`](results/figure2.pdf)

## Quick start

```bash
uv sync --group figures        # Python 3.12
.venv/bin/python -m pytest -q tests/test_verdicts.py tests/test_majority.py tests/test_judge.py tests/test_quickcheck.py
```

The full suite (`pytest -q`) takes about an hour, mostly `tests/test_unblind_all.py`; two launch-config tests skip without the run data.

## Repository map

| Path | What it is |
|---|---|
| `collusion/` | market, game loop and prompts, honeypot and self-play runners, model clients, analysis (scoring, judges, statistics) |
| `scripts/unblind_all.py` | the one-command analysis: checks label hashes, computes the tests, recomputes key statistics independently, writes `results/` |
| `scripts/` (other) | how runs were launched and judged (`launch_main.py`, `opus_cc*`), regression and validation checks |
| `configs/` | frozen run settings and the model ↔ run pairing |
| `docs/hand-labels/` | the author's human-check labels, judge gate results, label-file hash lists |
| `docs/` (other) | method notes: [prompts](docs/prompts.md), [judging](docs/judging.md), [Opus judge](docs/opus-judge.md), [analysis](docs/analysis-methods.md) |
| `PREREG.md` | the analysis plan and its log of changes |
| `results/` | the unblinded output |
| `tests/` | offline tests, including planted-truth tests of the analysis |

## Reproduce the analysis

The run data (several GB of logs, messages, prices and judge labels) are available from the author on request. Place them in `runs/` at the repository root and run `.venv/bin/python scripts/unblind_all.py --unblind --out results`. The human-check answer key is withheld so the check stays reusable; without it, the script uses the recorded κ values, and every headline number is unaffected. New runs need API keys (`.env.example`) and, for open-weight models, GPU pods; see `python -m collusion --help`.

## Data, integrity and prior work

- **Released here:** code, analysis plan, hand labels, label-file hash lists and analysis outputs. Run logs are on request; pilot and pre-sprint data are exploratory and not released.
- **Integrity:** this release has a fresh history. [`docs/integrity-log.md`](docs/integrity-log.md) lists the development-repository commits and tags the paper cites, with push times. The author's name was replaced by "the author", except in checksum-pinned files (hand-label sheet, gate reports, `results/RESULTS.json`), which are byte-identical and still contain the first name. Local paths in `results/` were made relative; no values changed.
- **Before the sprint:** the simulation harness, pilots and an exploratory self-play grid (21 Oct 2026). On 22 Oct the code moved into this layout without changing its behaviour (`scripts/regress.py`: identical outputs on 1,391 files), and the nine-model lineup and self-play setup were frozen.
- **During the sprint (23–25 Oct):** the honeypot design changes; every analysed run (self-play launched just after midnight on Friday 23 Oct, Berlin time, as the organisers allowed); the judges, hand labels, analysis plan, analysis and paper. The pre-sprint grid and pilots shaped the design but never enter the analysis.

## LLM usage

The author chose the question, made the research decisions, hand-labelled the human check and checked every claim and result. Claude Opus 5.5 (Anthropic), in Claude Code, wrote most of the code and drafted the paper text under the author's direction; it is also the words judge.

## Licence

MIT for the code. Third-party parts, including market code adapted from Peist, Romero and Sauer and prompt text reproduced from LLMCartel (Riemer et al., 2026), keep their original terms; see [`ATTRIBUTION.md`](ATTRIBUTION.md).
