"""Logit demand (Calvano et al. 2020), benchmarks and profit gain.

The demand and benchmark logic is adapted from algorithmic_pricing_llms
(MIT, Peist, Romero & Sauer 2025), simplified to plain logit demand
(no nesting). See ATTRIBUTION.md.

Demand for firm i at prices p:
    q_i = beta * exp((a_i - p_i/alpha)/mu) / (sum_j exp((a_j - p_j/alpha)/mu) + exp(a0/mu))
Profit (as in LLMCartel, Riemer et al. 2026):
    pi_i = (p_i - alpha*c_i) * q_i
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize, minimize_scalar


@dataclass(frozen=True)
class Market:
    a: tuple[float, ...]
    c: tuple[float, ...]
    mu: float = 0.25
    a0: float = 0.0
    alpha: float = 1.0
    beta: float = 100.0

    @property
    def n(self) -> int:
        return len(self.a)

    def quantities(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        a = np.asarray(self.a, dtype=float)
        x = (a - p / self.alpha) / self.mu
        x0 = self.a0 / self.mu
        m = max(float(x.max()), x0)  # stabilise the exponentials
        e = np.exp(x - m)
        return self.beta * e / (e.sum() + math.exp(x0 - m))

    def profits(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return (p - self.alpha * np.asarray(self.c, dtype=float)) * self.quantities(p)

    def best_response(self, i: int, p, upper: float) -> float:
        p = list(map(float, p))

        def neg_profit(x):
            p[i] = x
            return -self.profits(p)[i]

        lower = self.alpha * self.c[i]
        return float(minimize_scalar(neg_profit, bounds=(lower, upper), method="bounded",
                                     options={"xatol": 1e-12}).x)

    def monopoly_prices(self) -> np.ndarray:
        return _monopoly(self)

    def nash_prices(self) -> np.ndarray:
        return _nash(self)

    def pairwise_monopoly_prices(self) -> np.ndarray:
        """LLMCartel's 'monopoly' benchmark: the first-order condition of a firm that
        internalises ONE rival's profit. Equals the joint monopoly for a duopoly; for
        3+ firms it is lower than the true joint monopoly (1.925 vs 2.848 in the
        triopoly preset). Symmetric markets only."""
        if len(set(self.a)) != 1 or len(set(self.c)) != 1:
            raise ValueError("pairwise monopoly benchmark is defined for symmetric markets only")
        return _pairwise_monopoly(self)

    def benchmarks(self) -> dict:
        pn, pm = self.nash_prices(), self.monopoly_prices()
        out = {
            "nash_prices": pn.tolist(),
            "nash_profits": self.profits(pn).tolist(),
            "nash_quantities": self.quantities(pn).tolist(),
            "monopoly_prices": pm.tolist(),
            "monopoly_profits": self.profits(pm).tolist(),
            "monopoly_quantities": self.quantities(pm).tolist(),
        }
        if len(set(self.a)) == 1 and len(set(self.c)) == 1:
            pw = self.pairwise_monopoly_prices()
            out["pairwise_monopoly_prices"] = pw.tolist()
            out["pairwise_monopoly_profits"] = self.profits(pw).tolist()
        return out

    def profit_gain(self, profits, mode: str = "joint") -> np.ndarray:
        """Per-firm Delta = (pi - pi_Nash) / (pi_Monopoly - pi_Nash).
        mode='joint' uses the true joint monopoly; mode='pairwise' uses LLMCartel's benchmark."""
        b = self.benchmarks()
        pn = np.asarray(b["nash_profits"])
        pm = np.asarray(b["monopoly_profits"] if mode == "joint" else b["pairwise_monopoly_profits"])
        return (np.asarray(profits, dtype=float) - pn) / (pm - pn)


@lru_cache(maxsize=64)
def _monopoly(m: Market) -> np.ndarray:
    lower = [m.alpha * ci for ci in m.c]
    x0 = [lo + 0.5 * m.alpha for lo in lower]
    res = minimize(lambda p: -m.profits(p).sum(), x0, method="L-BFGS-B",
                   bounds=[(lo, None) for lo in lower], options={"ftol": 1e-15, "gtol": 1e-12})
    return np.asarray(res.x, dtype=float)


@lru_cache(maxsize=64)
def _pairwise_monopoly(m: Market) -> np.ndarray:
    from scipy.optimize import brentq

    c = m.alpha * m.c[0]

    def foc(p):
        share = m.quantities([p] * m.n)[0] / m.beta
        return 1 - (p - c) * (1 - share) / m.mu + (p - c) * share / m.mu

    p = brentq(foc, c + 1e-9, c + 50 * m.alpha, xtol=1e-14)
    return np.full(m.n, p)


@lru_cache(maxsize=64)
def _nash(m: Market, tol: float = 1e-11, max_steps: int = 2000) -> np.ndarray:
    # Fixed point of simultaneous best responses, started at marginal cost.
    upper = float(_monopoly(m).max()) * 2
    p = np.asarray([m.alpha * ci for ci in m.c], dtype=float) + 0.01
    for _ in range(max_steps):
        nxt = np.asarray([m.best_response(i, p, upper) for i in range(m.n)])
        if np.max(np.abs(nxt - p)) < tol:
            return nxt
        p = nxt
    raise RuntimeError("Nash fixed point did not converge")


PRESETS = {
    # Calvano et al. 2020 / Fish et al. 2024 / LLMCartel default duopoly.
    "duopoly": Market(a=(2.0, 2.0), c=(1.0, 1.0), mu=0.25),
    # LLMCartel 3-firm setting, chosen to keep Nash and monopoly prices close to the duopoly.
    "triopoly": Market(a=(3.0, 3.0, 3.0), c=(1.0, 1.0, 1.0), mu=0.315),
}
