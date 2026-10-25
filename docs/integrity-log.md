# Integrity log

The code in this repository was released with a fresh history. The paper cites commits and a tag from the private
development repository. This table lists each one with its commit time (self-reported by git) and the time GitHub
recorded its first push (from the repository's push events). The full development history is available from the
author on request.

| commit | what it records | commit time | first pushed to GitHub (UTC) | subject |
|---|---|---|---|---|
| `d005a6ee8e99` | analysis plan frozen (tag analysis-freeze) | 2026-10-25 11:25 +0100 | 2026-10-25 11:17:16 (branch honeypot-v2) | PREREG v5.4 FROZEN on the author's OK (tag analysis-freeze) |
| `34e1389b1586` | Opus judge plan committed, before it ran | 2026-10-25 12:24 +0100 | 2026-10-25 12:09:35 (branch honeypot-v2) | Opus 5.5 third words judge via Claude Code subagents (exploratory; prep, not run) |
| `a65ce83e8620` | quick-check gate and exploratory Opus judge defined, before any label | 2026-10-25 13:09 +0100 | 2026-10-25 12:09:35 (branch honeypot-v2) | PREREG v5.5: quick-check gate (explicit assent) and exploratory Opus 5.5 judge, before any label and before unblinding |
| `5a94afd319d3` | human-check gate result recorded | 2026-10-25 13:54 +0100 | 2026-10-25 12:54:10 (branch honeypot-v2) | Hand-label gate result recorded before unblinding: kappa 0.386 < 0.6, H1 words claims on hold (Opus vs the author 0.682, exploratory) |
| `03b5627bee29` | Opus chosen as the interpreted words judge | 2026-10-25 14:39 +0100 | 2026-10-25 13:39:39 (branch honeypot-v2) | PREREG: Opus 5.5 is the interpreted words judge (decided after the gate, before unblinding); Opus run record |
| `26fc5bf77d28` | known-answer tests for the Opus judge | 2026-10-25 14:45 +0100 | 2026-10-25 15:50:39 (branch honeypot-v2) | opus_validate: known-answer fake chats for the Opus subagent judge (dump + scorer) and pooled, model-free laziness diagnostics |
| `17391a3c8530` | reporting rule recorded | 2026-10-25 14:58 +0100 | 2026-10-25 14:02:44 (branch honeypot-v2) | Before unblinding: Opus in the main text, registered judges in an appendix; one clause if their H1 conclusion differs |
| `8c9926cfaf6c` | label-file hashes, list 1 | 2026-10-25 14:57 +0100 | 2026-10-25 13:57:20 (branch honeypot-v2) | Before unblinding: hashes of all judge label files; commit to report registered and Opus H1 side by side |
| `ba1e2fec2734` | majority of three Opus runs made primary | 2026-10-25 15:02 +0100 | 2026-10-25 14:02:44 (branch honeypot-v2) | Before unblinding: primary words judge = majority of three Opus runs (fallback run 1 if it misses the human bar) |
| `8815099c8cac` | different-conclusion clause made precise | 2026-10-25 15:09 +0100 | 2026-10-25 14:09:32 (branch honeypot-v2) | Before unblinding: make the 'different conclusion' clause rule precise |
| `57859af201f9` | Opus run 3 extended to all items | 2026-10-25 15:21 +0100 | 2026-10-25 14:21:56 (branch honeypot-v2) | Before unblinding: V3 labels all items (selective V3 not equivalent); kappa intervals recorded |
| `71368c606c76` | all Opus runs complete; label-file hashes, list 2 | 2026-10-25 16:16 +0100 | 2026-10-25 15:16:20 (branch honeypot-v2) | Before unblinding: Opus V2/V3 complete; opus-maj3 passes the human check (kappa 0.682) and is primary; run agreement; label hashes 2 |
| `f40fed72b564` | unblinding script used for the results | 2026-10-25 20:58 +0100 | 2026-10-25 21:23:45 (branch honeypot-v2) | Merge branch 'unblind-fix' into honeypot-v2 |
| `d382965c8d54` | unblinded results committed | 2026-10-25 22:23 +0100 | 2026-10-25 21:23:45 (branch honeypot-v2) | Unblinded results (unblind_all.py at f40fed7; all verification checks passed) |
| `508173ce3e3c` | pre-sprint package moved into this code layout (22 Oct) | 2026-10-22 00:01 +0200 | 2026-10-21 22:29:54 (branch main) | Modular package moved from the pre-sprint harness (apart-collusion-sprint, harness/bertrand @ 24d02b5) |
| `b3739d65b6b2` | Python pinned; regression check against the pre-sprint harness | 2026-10-22 00:25 +0200 | 2026-10-21 22:29:54 (branch main) | Pin Python 3.12 (all pre-sprint runs used 3.12.12; uv had picked 3.14). Tests 92 pass; regress vs old: 0 differences |

The tag `analysis-freeze` points to `d005a6e`. Times are in the committer's local zone (+0100 = CET, +0200 = CEST).
