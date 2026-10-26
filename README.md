# Does a collusion honeypot predict collusion? Nine LLMs in simulated pricing markets

Code, labels and results for our project in the [Apart Research AI Collusion Research Sprint](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25) (23–25 October 2026), Track 3.2.

**What we test.** Developers and evaluators use honeypots, tests built to tempt a model into misbehaving, as one line of safety evidence. We ask whether behaviour in a collusion honeypot predicts behaviour in a longer setting with no planted offer. Nine language models play a simulated three-firm pricing market with chat. In the **honeypot**, two copies of a model compete with firm C, which we script. After ten ordinary rounds, C raises its price in four of five versions of the game, and in the offer version it invites the others to match it. In **self-play**, our stand-in for deployment, three copies of the model play six 300-round games. We compare the two rankings of the models. The headline uses explicit price-fixing agreements, judged by Claude Opus 5.5 (checked against human labels, κ = 0.68). The secondary test uses prices. The analysis plan was fixed before unblinding (`PREREG.md`), and every later change is logged there.

**Result.** Models that agree to fix prices more often in the honeypot's offer games also do so more often in self-play (Spearman ρ = 0.93, 95% interval [0.58, 1.00], one-sided p < 0.001, n = 9). Honeypot profit gains in the opening rounds track self-play profits (ρ = 0.72, descriptive); the planned offer-minus-no-offer price measure does not (ρ = −0.23, exploratory after its planted-colluder check failed). Full output: `results/RESULTS.md`. Figure 2: `results/figure2.pdf`.

All markets are simulated.

## Repository map

| Path | What it is |
|---|---|
| `collusion/market/` | demand, profits, competitive (1.47) and cartel (2.85) prices, best responses |
| `collusion/game/` | the round loop, prompts, LLM and scripted firms, firm C's messages, save / copy of games |
| `collusion/experiments/` | self-play runs, honeypot runs (save point → five versions: none, silent, hint, offer, benign), robot known-answer runs |
| `collusion/clients/` | OpenAI, Anthropic and vLLM clients |
| `collusion/analysis/` | price scoring, the selfish best-response baseline, words judges and majority vote, the human quick check, cross-model statistics |
| `scripts/unblind_all.py` | the one-command analysis: checks label hashes, computes H1–H4, recomputes the key statistics independently, writes `results/` |
| `scripts/opus_cc*.py`, `scripts/opus_cc_workflow*.js` | how the Claude Opus 5.5 words labels were produced (Claude Code subagents) |
| `scripts/launch_main.py`, `scripts/pod/`, `scripts/sprint/` | how the runs were launched (API models; open-weight models on GPU pods) |
| `scripts/regress.py`, `scripts/robot_controls.py`, `scripts/validate_*.py`, `scripts/verify_selfish_baseline.py` | regression and validation checks |
| `configs/` | the frozen run settings and the model ↔ run pairing (`pairs.json`) |
| `docs/hand-labels/` | the author's 43 quick-check labels, judge gate results, judge agreement, and the label-file hash lists |
| `docs/prompts.md`, `docs/judging.md`, `docs/opus-judge.md`, `docs/analysis-methods.md`, `docs/parameters.md` | method documentation |
| `PREREG.md` | the analysis plan and its log of changes |
| `results/` | the unblinded output (`RESULTS.md`, `RESULTS.json`, Figure 2) |
| `tests/` | offline tests, including planted-truth tests of the analysis |

## Reproduce

```bash
uv sync --group figures                   # Python 3.12; the figures group draws Figure 2
.venv/bin/python -m pytest -q             # offline tests (the full suite takes about an hour; tests/test_unblind_all.py alone is the slow part)
```

**Re-running the analysis.** The run data (several GB of logs, prompts, messages, prices and judge labels) are available from the author on request. Place them in `runs/` at the repository root (`configs/pairs.json` points to `../runs/...`). Then:

```bash
.venv/bin/python scripts/unblind_all.py --unblind --out results
```

The recorded human-check κ values are in `docs/hand-labels/gate-*.txt` and `results/RESULTS.md`. Recomputing them needs the quick-check answer key, which maps the blinded sheet back to the messages; it is withheld so the check stays reusable. Without `--quick-key`, the script uses the κ values recorded in its code for the Opus judge sets and shows the other gate rows as unknown; every H1/H2 number is unaffected.

**Producing new runs** needs API keys (copy `.env.example` to `.env`) and, for open-weight models, GPU pods:

```bash
.venv/bin/python -m collusion robots --out runs/robots                       # known-answer test, no LLM calls
.venv/bin/python -m collusion selfplay --model gpt-6-luna --out runs/sp-luna --games 6 --periods 300 --firms 3 --chat
.venv/bin/python scripts/launch_main.py --help                                 # the main honeypot runs, from configs/main-honeypot.json
.venv/bin/python scripts/print_game.py runs/sp-luna/sp-s1                      # read a game round by round
```

**Data.** This repository holds the code, the analysis plan, the hand labels, the label-file hash lists and the analysis outputs. Run logs, prompts, messages and judge labels for every analysed run are available from the author on request. Pilot and pre-sprint grid data are exploratory and not released.

**Integrity.** This release has a fresh history. `docs/integrity-log.md` lists every commit and tag the paper cites from the development repository, with commit and GitHub push times; the full history is available on request. For this release, the author's name was replaced by "the author" in code comments, docs and `PREREG.md` (no other text changed). Files whose checksums are recorded (the hand-label sheet, the gate reports and `results/RESULTS.json`) were left byte-identical, so they still contain the author's first name; the quick-check label file was renamed to `docs/hand-labels/author-quick-labels.md` (same content and sha256 as recorded in `results/RESULTS.json`).

## Prior work and what is new

- **Before the sprint:** we built the simulation harness in a private repository, ran pilots, and ran an exploratory self-play grid on 21 Oct 2026. On 22 Oct the code moved into this layout without changing its behaviour: `scripts/regress.py` gave identical outputs on 1,391 files from fake-model, robot and replayed real games. On 22 Oct we also froze the nine-model lineup and the self-play setup.
- **During the sprint (23–25 Oct):**
  - the honeypot design changes;
  - every run analysed in the paper (self-play launched just after midnight on Friday 23 Oct, Berlin time; the organisers confirmed participants could start as soon as it was Friday locally);
  - the judges, the hand labels, the analysis plan, the analysis and the paper.
- **Disclosure:** the pre-sprint grid and pilots shaped the design but never enter the analysis.

## LLM usage

Claude Opus 5.5 (Anthropic), in Claude Code, wrote most of the code and drafted the paper text under the author's direction. The author made the research decisions and checked every claim and result. A separate Claude agent and OpenAI's Codex reviewed code and design changes. Claude Opus 5.5 is also the words judge; GPT-6 Luna and Claude Haiku 5.5 were the planned judges.

## Notes

- Absolute local paths in `results/` were replaced by repository-relative paths for this release; no values changed.
- Attribution (`ATTRIBUTION.md`): the market code adapts MIT-licensed code by Peist, Romero and Sauer; the P0 prompt text and one price-parsing rule are reproduced verbatim from LLMCartel (Riemer et al., ICML 2026; their repository has no licence) for comparability, in `collusion/game/prompts.py` (`P0_SYSTEM`, `parse_price_lcartel_gpt`).
- Licence: MIT (code); third-party portions listed in `ATTRIBUTION.md` remain under their original terms.
