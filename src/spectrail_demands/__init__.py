"""Demand generator of the spectrail-rsa-episodes dataset.

Draws dynamic RSA demand sequences from the same distributions as the
``demands50`` and ``demands100`` traffic sets, on the same topologies::

    from spectrail_demands import generate_sequence, demand_stream

    seq = generate_sequence("Iinet", "constanterlang", seed=1_000_001, traffic_set="demands50")
    for d in demand_stream("Iinet", "oscillatingarrival", seed=7, traffic_set="demands100"):
        ...
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from spectrail_demands.generate import PATTERNS, Demand, mouseelephant_probabilities, stream, take
from spectrail_demands.parameters import (
    ROLES,
    TRAFFIC_SETS,
    TrafficParameters,
    get_parameters,
    load_topology,
    topologies,
)

__version__ = "0.1.0"

__all__ = [
    "PATTERNS",
    "ROLES",
    "TRAFFIC_SETS",
    "Demand",
    "TrafficParameters",
    "demand_stream",
    "generate_sequence",
    "get_parameters",
    "load_topology",
    "mouseelephant_probabilities",
    "topologies",
    "to_arrays",
]


def demand_stream(
    topology: str,
    pattern: str,
    seed: int | np.random.Generator,
    *,
    traffic_set: str,
) -> Iterator[Demand]:
    """Infinite demand stream with the released parameters of (traffic_set, topology, pattern)."""
    p = get_parameters(traffic_set, topology, pattern)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    nodes = list(load_topology(topology).nodes())
    return stream(pattern, nodes, p.demandvalues, p.interarrivaltime, p.servicetime, rng)


def generate_sequence(
    topology: str,
    pattern: str,
    seed: int | np.random.Generator,
    *,
    traffic_set: str,
    n: int | None = None,
) -> list[Demand]:
    """One sequence of ``n`` demands (default: the released ``demandnum``)."""
    n = get_parameters(traffic_set, topology, pattern).demandnum if n is None else n
    return take(demand_stream(topology, pattern, seed, traffic_set=traffic_set), n)


def to_arrays(demands: list[Demand]) -> dict[str, np.ndarray]:
    """Column arrays (source, target, demandvalue, tin, tout)."""
    return {
        "source": np.fromiter((d.source for d in demands), np.int32, len(demands)),
        "target": np.fromiter((d.target for d in demands), np.int32, len(demands)),
        "demandvalue": np.fromiter((d.demandvalue for d in demands), np.int8, len(demands)),
        "tin": np.fromiter((d.tin for d in demands), np.float64, len(demands)),
        "tout": np.fromiter((d.tout for d in demands), np.float64, len(demands)),
    }
