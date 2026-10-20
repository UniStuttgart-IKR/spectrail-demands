"""Derive generator parameters for a new topology.

Ports of ``IKRNetBase.get_avamax_servicetime`` and
``IKRNetBase.get_almost_steadystate_demandnumber``. The released parameter
table was computed with these rules; use them only for graphs outside it.
``derive_parameters`` reproduces every released ``servicetime`` exactly.
The demand count comes from a seeded Monte Carlo estimate, so it differs from
the Julia value by a few percent (the RNG streams differ).
"""

from __future__ import annotations

from collections.abc import Sequence
import bisect

import networkx as nx
import numpy as np

DEFAULT_DEMAND_VALUES = (2, 3, 4, 5)
DEFAULT_SPECTRUM_SLOTS = 50
DEFAULT_INTERARRIVAL = 1.0
STDMULT = {"constanterlang": 1.0, "mouseelephant": 1.0, "oscillatingarrival": 0.7}


def avamax_servicetime(
    G: nx.DiGraph,
    demandvalues: Sequence[int] = DEFAULT_DEMAND_VALUES,
    spectrumslotnumber: int = DEFAULT_SPECTRUM_SLOTS,
    stdmult: float = 1.0,
) -> float:
    """Mean holding time that fills the spectrum on average (interarrival time 1).

    ``slots * directed_edges / (mean_hops - s*std_hops) / (mean_width - s*std_width)``
    over all unordered node pairs, sample standard deviations, s = stdmult.
    """
    nodes = sorted(G.nodes())
    dists = []
    for i, u in enumerate(nodes):
        lengths = nx.single_source_shortest_path_length(G, u)
        dists.extend(float(lengths.get(nodes[j], np.inf)) for j in range(i))
    mean_p = float(np.mean(dists))
    std_p = float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0
    mean_d = float(np.mean(demandvalues))
    std_d = float(np.std(demandvalues, ddof=1))
    # Julia guard against a vanishing denominator.
    smax = min(mean_p / std_p if std_p else np.inf, mean_d / std_d if std_d else np.inf)
    if abs(stdmult - smax) < 0.001:
        stdmult = 0.95 * smax
    return (
        spectrumslotnumber * G.number_of_edges()
        / (mean_p - stdmult * std_p) / (mean_d - stdmult * std_d)
    )


def steadystate_demandnumber(
    G: nx.DiGraph,
    service: float,
    demandvalues: Sequence[int] = DEFAULT_DEMAND_VALUES,
    spectrumslotnumber: int = DEFAULT_SPECTRUM_SLOTS,
    interarrival: float = DEFAULT_INTERARRIVAL,
    rng: np.random.Generator | None = None,
    multiplier: int = 10,
    iterations: int = 100,
) -> int:
    """Heuristic sequence length: ``multiplier`` times the largest number of
    arrivals, over ``iterations`` M/M/inf runs, until the number of demands in
    the system first reaches ``avamax_servicetime(stdmult=0)``."""
    rng = np.random.default_rng(0) if rng is None else rng
    buffer = avamax_servicetime(G, demandvalues, spectrumslotnumber, stdmult=0.0)
    best = -1
    for _ in range(iterations):
        count, active, t, expiry = 0, 0, 0.0, []
        while active < buffer:
            while expiry and expiry[0] < t:
                expiry.pop(0)
                active -= 1
            t += rng.exponential(interarrival)
            count += 1
            active += 1
            bisect.insort(expiry, t + rng.exponential(service))
        best = max(best, count)
    return best * multiplier


def derive_parameters(
    G: nx.DiGraph,
    pattern: str,
    demandvalues: Sequence[int] = DEFAULT_DEMAND_VALUES,
    spectrumslotnumber: int = DEFAULT_SPECTRUM_SLOTS,
    interarrival: float = DEFAULT_INTERARRIVAL,
) -> tuple[float, int]:
    """``(servicetime, demandnum)`` for ``pattern`` on ``G``."""
    service = avamax_servicetime(G, demandvalues, spectrumslotnumber, STDMULT[pattern])
    n = steadystate_demandnumber(G, service, demandvalues, spectrumslotnumber, interarrival)
    return service, n
