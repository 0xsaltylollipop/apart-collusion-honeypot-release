"""Self-play grid table (21 Oct): Delta (last 50 rounds) per model x scenario, at each game's final round and at a
common horizon, with per-game values, rounds reached and the settled flag (each firm within 5% of its mean over the
last 100 rounds). Delta is in each market's own scale (2-firm: Nash 1.473 / monopoly 1.925; 3-firm: 1.472 / 2.848);
never pool 2- and 3-firm numbers.

    .venv/bin/python scripts/selfplay_table.py [--horizon-chat 300] [--horizon-nochat N] > table.md
    .venv/bin/python scripts/selfplay_table.py runs/<run>/selfplay/f3-chat ... [--upto N] [--threshold 0.5]
        (any self-play dirs, one row each: per-game Delta, mean, SD, at-cap share, high-Delta k of n)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.analysis.scoring import selfplay_stats  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"
MODELS = ["oss", "r1q32", "gemma", "q38", "muse", "nemo", "luna"]
SCEN = ["f2-chat", "f3-chat", "f2-nochat", "f3-nochat"]


def cell(d: Path, upto: int | None) -> str:
    s = selfplay_stats(d, upto=upto)
    games = list(s["games"].values())
    if not games or s["late"] is None:
        return "–"
    per = "/".join(f"{g['late']:.2f}" if g["late"] is not None else "–" for g in games)
    settled = sum(bool(g.get("settled")) for g in games)
    short = "*" if upto and any(g["periods"] < upto for g in games) else ""
    return f"**{s['late']:.2f}**{short} ({per}) s{settled}/{len(games)}"


def rounds(d: Path) -> list[int]:
    s = selfplay_stats(d)
    return [g["periods"] for g in s["games"].values()]


def table(dirs: list[Path], upto: int | None, threshold: float | None):
    print("| run | Delta per game | mean | SD | at cap (mean share) | high-Delta k/n | rounds |\n|---|---|---|---|---|---|---|")
    for d in dirs:
        s = selfplay_stats(d, upto=upto, threshold=threshold)
        games = list(s["games"].values())
        if not games:
            print(f"| {d} | – | – | – | – | – | – |")
            continue
        per = "/".join(f"{g['late']:.2f}" for g in games)
        sd = "–" if s["sd"] is None else f"{s['sd']:.2f}"
        cap = sum(g["at_cap"] for g in games) / len(games)
        hd = "–" if s["high_delta"] is None else f"{s['high_delta']['k']}/{s['high_delta']['n']}"
        prov = " (provisional)" if s["provisional"] else ""
        print(f"| {d} | {per} | **{s['late']:.2f}**{prov} | {sd} | {cap:.2f} | {hd} | "
              f"{'/'.join(str(g['periods']) for g in games)} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", type=Path, help="self-play dirs (default: the 21 Oct grid in runs/sp-grid-*)")
    ap.add_argument("--upto", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--horizon-chat", type=int, default=300)
    ap.add_argument("--horizon-nochat", type=int, default=None, help="default: the fewest rounds any game reached")
    a = ap.parse_args()
    if a.dirs:
        table(a.dirs, a.upto, a.threshold)
        return
    dirs = {(m, sc): RUNS / f"sp-grid-{m}" / "selfplay" / sc for m in MODELS for sc in SCEN}
    dirs = {k: v for k, v in dirs.items() if (v / "manifest.json").exists()}
    if not dirs:
        raise SystemExit(f"no 21 Oct grid runs under {RUNS} (they live in the archive repo); pass self-play dirs instead")
    reached = {k: rounds(v) for k, v in dirs.items()}
    h_nochat = a.horizon_nochat or min(min(r) for (m, sc), r in reached.items() if "nochat" in sc)
    print(f"# Self-play grid: Delta over the last 50 rounds (0 = Nash, 1 = joint monopoly)\n")
    print("Cell: **scenario mean** (game 1/2/3) s<settled games>/<games>; * = some game stopped before the horizon "
          "(its own last 50 rounds are used). Never pool 2- and 3-firm columns.\n")
    print("## Rounds reached (per game)\n")
    print("| model | " + " | ".join(SCEN) + " |\n|---|" + "---|" * len(SCEN))
    for m in MODELS:
        print(f"| {m} | " + " | ".join(str(reached.get((m, sc), "–")) for sc in SCEN) + " |")
    for title, upto in (("At each game's final round", None),
                        (f"Common horizon: chat at round {a.horizon_chat}, no chat at round {h_nochat}", "h")):
        print(f"\n## {title}\n")
        print("| model | " + " | ".join(SCEN) + " |\n|---|" + "---|" * len(SCEN))
        for m in MODELS:
            row = []
            for sc in SCEN:
                if (m, sc) not in dirs:
                    row.append("–")
                    continue
                u = None if upto is None else (h_nochat if "nochat" in sc else a.horizon_chat)
                row.append(cell(dirs[(m, sc)], u))
            print(f"| {m} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
