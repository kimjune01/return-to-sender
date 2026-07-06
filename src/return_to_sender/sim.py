"""Nodewise e-values under graph interference, on synthetic agent-email graphs.

Implements the five-step composition from the Return to Sender post:
  1. Estimand: per-node effect of throttling on own-neighborhood harm,
     neighbors' treatments integrated over the reference design.
  2. Nodewise Horvitz-Thompson scores under Bernoulli randomization.
  3. Scores accumulate across decision windows.
  4. A betting supermartingale per node turns the running scores into an
     anytime-valid e-value (mixture over a bet-size grid).
  5. e-BH across nodes flags the abusive set with FDR control at level q.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class World:
    """A synthetic agent-email platform: graph, roles, harm parameters."""

    n_agents: int = 400
    mean_degree: float = 8.0
    n_relays: int = 20
    n_subcontractors: int = 20  # degree-matched honest forwarders, delta = 0
    beta: float = 3.0  # harm a relay deposits in its neighborhood per active window
    background: float = 0.05  # per-neighbor rate of organic (non-causal) harm
    adjacency: np.ndarray = field(init=False)
    relays: np.ndarray = field(init=False)
    subcontractors: np.ndarray = field(init=False)

    def build(self, rng: np.random.Generator) -> "World":
        n = self.n_agents
        p_edge = self.mean_degree / (n - 1)
        upper = rng.random((n, n)) < p_edge
        adj = np.triu(upper, 1)
        self.adjacency = (adj | adj.T).astype(np.int8)
        # Relays and subcontractors drawn from the same degree distribution:
        # identical edge structure, only the harm parameter differs.
        degrees = self.adjacency.sum(axis=1)
        order = np.argsort(degrees)
        picks = rng.permutation(order[n // 4 : 3 * n // 4])  # mid-degree band
        self.relays = np.zeros(n, dtype=bool)
        self.relays[picks[: self.n_relays]] = True
        self.subcontractors = np.zeros(n, dtype=bool)
        self.subcontractors[picks[self.n_relays : self.n_relays + self.n_subcontractors]] = True
        return self

    def simulate_windows(
        self, treatments: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Harm observed in each node's closed neighborhood, per window.

        treatments: (n, t) binary, 1 = throttled.
        Returns (n, t) harm counts. A relay contributes beta to every
        closed-neighborhood it belongs to only in windows it is active.
        """
        n, _ = treatments.shape
        closed = self.adjacency + np.eye(n, dtype=np.int8)
        active_harm = (1 - treatments) * self.relays[:, None] * self.beta
        harm = closed @ active_harm
        noise = rng.binomial(closed.sum(axis=1)[:, None], self.background, harm.shape)
        return harm + noise



def ht_scores(harm: np.ndarray, treatments: np.ndarray, p: float) -> np.ndarray:
    """Horvitz-Thompson score per node per window.

    Unbiased for E[Y | v active] - E[Y | v throttled] under independent
    Bernoulli(p) throttling; neighbors' treatments integrate out.
    """
    return harm * ((1 - treatments) / (1 - p) - treatments / p)


def aipw_scores(harm: np.ndarray, treatments: np.ndarray, p: float) -> np.ndarray:
    """Doubly robust (AIPW) score per node per window.

    Outcome model: per-node, per-arm running means over past windows only,
    so the score at window t is measurable given the past and stays
    conditionally unbiased for delta_v under the design regardless of how
    well the means fit. The residual terms shrink as the fit improves.
    """
    n, t_max = harm.shape
    scores = np.empty_like(harm, dtype=float)
    sums = np.zeros((n, 2))
    counts = np.zeros((n, 2))
    for t in range(t_max):
        with np.errstate(invalid="ignore"):
            m0 = np.where(counts[:, 0] > 0, sums[:, 0] / np.maximum(counts[:, 0], 1), 0.0)
            m1 = np.where(counts[:, 1] > 0, sums[:, 1] / np.maximum(counts[:, 1], 1), 0.0)
        z = treatments[:, t]
        y = harm[:, t]
        scores[:, t] = (
            m0 - m1 + (1 - z) * (y - m0) / (1 - p) - z * (y - m1) / p
        )
        sums[np.arange(n), z] += y
        counts[np.arange(n), z] += 1
    return scores


def e_process(
    scores: np.ndarray,
    tau: float,
    score_floor: float,
    fractions: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8),
    cap: float = 1e250,
) -> np.ndarray:
    """Betting supermartingale per node, mixed over a bet-size grid.

    Tests H0: delta_v <= tau. Each bet multiplies wealth by
    1 + lambda * (S_t - tau); lambda is capped so wealth stays nonnegative
    at the worst realizable score. A uniform mixture of e-processes is an
    e-process. Returns the terminal e-value per node.
    """
    n, _ = scores.shape
    lam_max = 1.0 / (tau - score_floor)  # keeps 1 + lam*(S - tau) >= 0
    wealth = np.zeros(n)
    for frac in fractions:
        lam = frac * lam_max
        factors = 1.0 + lam * (scores - tau)
        factors = np.maximum(factors, 0.0)
        logs = np.log(np.maximum(factors, 1e-300)).sum(axis=1)
        wealth += np.exp(np.minimum(logs, np.log(cap)))
    return wealth / len(fractions)


def e_bh(e_values: np.ndarray, q: float) -> np.ndarray:
    """e-BH: flag the k largest e-values where e_(k) >= n / (q * k)."""
    n = len(e_values)
    order = np.argsort(e_values)[::-1]
    ranked = e_values[order]
    ks = np.arange(1, n + 1)
    passing = ranked >= n / (q * ks)
    flagged = np.zeros(n, dtype=bool)
    if passing.any():
        k_star = np.max(ks[passing])
        flagged[order[:k_star]] = True
    return flagged


def run_trial(
    world: World,
    n_windows: int,
    p_throttle: float,
    tau: float,
    q: float,
    rng: np.random.Generator,
    confounded: bool = False,
    harm_cap: float = 6.0,
    estimator: str = "aipw",
) -> dict:
    """One full platform run: randomize, observe, accumulate, flag."""
    n = world.n_agents
    if confounded:
        # Observational regime: the platform throttles reactively, nodes
        # with harmful neighborhoods last window get throttled this window.
        # The analyst still assumes Bernoulli(p_throttle).
        treatments = np.zeros((n, n_windows), dtype=np.int8)
        z = rng.binomial(1, p_throttle, n)
        for t in range(n_windows):
            treatments[:, t] = z
            harm_t = world.simulate_windows(z[:, None], rng)[:, 0]
            hot = harm_t > np.quantile(harm_t, 1 - p_throttle)
            z = hot.astype(np.int8)
        harm = world.simulate_windows(treatments, rng)
    else:
        treatments = rng.binomial(1, p_throttle, (n, n_windows)).astype(np.int8)
        harm = world.simulate_windows(treatments, rng)

    # Analyst-side truncation: clip observed harm at a known cap. The tested
    # estimand is the clipped-harm effect; the clip buys a tight betting
    # floor, and a relay's per-window deposit sits well under the cap.
    harm_obs = np.minimum(harm, harm_cap)
    if estimator == "aipw":
        scores = aipw_scores(harm_obs, treatments, p_throttle)
    else:
        scores = ht_scores(harm_obs, treatments, p_throttle)
    score_floor = -harm_cap / min(p_throttle, 1 - p_throttle)
    e_values = e_process(scores, tau, score_floor)
    flagged = e_bh(e_values, q)

    true_pos = int((flagged & world.relays).sum())
    false_pos = int((flagged & ~world.relays).sum())
    n_flagged = int(flagged.sum())
    return {
        "fdp": false_pos / max(n_flagged, 1),
        "power": true_pos / world.relays.sum(),
        "n_flagged": n_flagged,
        "subcontractors_flagged": int((flagged & world.subcontractors).sum()),
    }
