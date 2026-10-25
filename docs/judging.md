# How Apart judges (checked 22 Oct 2026)

Sources: the AI Collusion sprint page (Guidelines, Overview, Resources tabs) and the judges' reviews of all 35 ranked
top-5 projects from the 7 ranked 2026 sprints (https://apartresearch.com/sprints/all).

## Rubric (3 dimensions, each 1–5, no weights)

| | Impact Potential & Innovation | Execution Quality | Presentation & Clarity |
|---|---|---|---|
| 3 | clear problem, some novelty | solid for the timeframe, limitations acknowledged | problem, method, findings, limits easy to extract |
| 4 | original approach or neglected area | thorough, convincing validation, results support conclusions | easy to follow, right level of detail |
| 5 | genuinely novel, clear theory of change | ambitious and rigorous; unusually robust validation | complex ideas made accessible |

## Format

- PDF on the official template (Guidelines tab). Max 8 pages, not counting references and appendices.
- Abstract: 150–250 words.
- Required appendix: "Limitations and Dual-Use / Ethical Considerations". It says what the result does and doesn't establish, and how the setup or findings could make real-world collusion easier or harder to detect.
- Report variance, seeds and baselines. "Always include the positive control." The method must be replicable: models, prompts, seeds, metrics.
- AI use is allowed, including for writing. The team is responsible for everything; made-up results or references disqualify.
- Prior work is allowed if disclosed. Mark pre-sprint code as preexisting in the README and the report.
- Video: optional (3–5 min). No judge mentions one in the 35 reviews.

## Q3.2: the organisers' suggested project (Resources tab)

- Use an explicit alliance condition as the **positive control**.
- Use a scripted player that makes the pact offer.
- Compare a **direct offer, an indirect hint and a benign cooperation request**. We have offer and hint; benign request is not built (decide in the honeypot session).

## What judges reward and punish (counts out of 35)

| Rewarded | Punished |
|---|---|
| honest calibration: failures, negatives, rejected designs reported (~16) | small n, single seed / model, underpowered (~17) |
| clarity, own voice (~12) | overclaiming (~13) |
| controls and baselines; "the sham control is the star" (~10) | confounds, untested alternative explanations (~9) |
| pre-registration (6), as long as it is kept | LLM judge not validated against humans: "a big problem" (~8) |
| checkable numbers (`verify.py`), real-world grounding (~4 each) | inconsistent numbers, missing figures, dead links (~7) |

No review mentions pre-sprint prep. Disclosed AI drafting was accepted.

## Checklist for our paper

1. One sharp claim; everything else goes in a side box or the appendix.
2. Freeze before data, and say so: primary metric and test, the high-Δ convention, the judge's model, prompt and rule.
3. Put the positive control and the no-trap / silent baselines up front.
4. Report κ for the words judge (we hand-label 40 items on Friday).
5. Variance everywhere: game dots, SD, CIs. Never headline a single run.
6. State the power up front (ρ across ~9 models).
7. Name the confounds:
   - Haiku follows any raise;
   - prices anchor on the $2.89 shown in the prompt (at-cap);
   - API vs vLLM sampling;
   - effort calibrated differently per model.
8. Use no overclaiming words; tie each claim to a figure.
9. Give the theory of change: pre-deployment screening.
10. One script recomputes every number from the logs; all links work.
11. Appendix: failed and rejected designs, exact prompts and C's brief, example transcripts.
12. A disclosure table (pre-sprint vs sprint) and a one-line note on AI use.
