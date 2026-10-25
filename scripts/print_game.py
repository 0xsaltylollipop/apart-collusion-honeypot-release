"""Print one self-play game round by round, for reading by eye: prices, the round's collusion score (mean per-firm
profit gain, Nash = 0, joint monopoly = 1, in the game's own market) and every message.

    .venv/bin/python scripts/print_game.py runs/<run>/selfplay/f3-chat/sp-s101 [--from 0 --to 300]
        [--no-messages] [--reasoning 0]   # --reasoning N: also print the first N characters of each firm's reasoning
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.experiments.selfplay import selfplay_market  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--from", dest="lo", type=int, default=0)
    ap.add_argument("--to", dest="hi", type=int, default=10**6)
    ap.add_argument("--no-messages", action="store_true")
    ap.add_argument("--reasoning", type=int, default=0)
    a = ap.parse_args()
    g = Path(a.game)
    rows = [json.loads(x) for x in open(g / "rounds.jsonl")]
    n = len(rows[0]["prices"])
    b = selfplay_market(n).benchmarks()
    pn, pm = b["nash_profits"][0], b["monopoly_profits"][0]
    calls = {}
    if a.reasoning:
        for line in open(g / "calls.jsonl"):
            c = json.loads(line)
            calls[(c["period"], c["firm"])] = (c.get("reasoning") or c.get("content") or "")[: a.reasoning]
    print(f"{g}: {n} firms, Nash price {b['nash_prices'][0]:.3f}, joint-monopoly price {b['monopoly_prices'][0]:.3f}")
    for r in rows:
        t = r["period"]
        if not a.lo <= t <= a.hi:
            continue
        d = sum((x - pn) / (pm - pn) for x in r["profits"]) / n
        flags = ("" if not any(r["parse_failed"]) else " PARSE-FAIL") + ("" if not any(r.get("truncated") or [])
                                                                        else " TRUNCATED")
        print(f"R{t + 1:>3} prices {' '.join(f'{p:.2f}' for p in r['prices'])}  D {d:+.2f}{flags}")
        if not a.no_messages:
            for i, m in enumerate(r.get("sent") or []):
                if m:
                    print(f"      F{i}: {m}")
        for i in range(n):
            if (t, i) in calls:
                print(f"      F{i} thinks: {calls[(t, i)]!r}")


if __name__ == "__main__":
    main()
