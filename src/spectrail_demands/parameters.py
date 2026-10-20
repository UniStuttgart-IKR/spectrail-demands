"""Released topologies and per-(traffic set, topology, pattern) generator parameters.

``data/parameters.csv`` holds the generator parameters of every traffic set,
topology and pattern (identical for all sequences of a cell). The
``reference`` and ``deeprmsa`` values equal the ``demand_sequences`` table of
spectrail-rsa-episodes v2.3.0. The ``large`` topologies (``demands50`` only)
have no released episodes. ``data/topologies/*.graphml`` holds the graphs as
stored in the ``graphs`` table of the ``id0`` database file: directed, every
link as two edges, GraphML node ids 0..N-1 equal to the demand ``source`` and
``target``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache
from importlib import resources

import networkx as nx

TRAFFIC_SETS = ("demands50", "demands100")
ROLES = ("reference", "deeprmsa", "large")


@dataclass(frozen=True)
class TrafficParameters:
    traffic_set: str
    topology: str
    role: str
    pattern: str
    demandnum: int
    servicetime: float
    interarrivaltime: float
    spectrumslotnumber: int
    demandvalues: tuple[int, ...]

    @property
    def offered_load(self) -> float:
        """Mean number of demands in the system (Erlang), servicetime / interarrivaltime."""
        return self.servicetime / self.interarrivaltime


@cache
def _table() -> dict[tuple[str, str, str], TrafficParameters]:
    text = resources.files(__package__).joinpath("data/parameters.csv").read_text()
    out = {}
    for r in csv.DictReader(text.splitlines()):
        p = TrafficParameters(
            traffic_set=r["traffic_set"],
            topology=r["topology"],
            role=r["role"],
            pattern=r["pattern"],
            demandnum=int(r["demandnum"]),
            servicetime=float(r["servicetime"]),
            interarrivaltime=float(r["interarrivaltime"]),
            spectrumslotnumber=int(r["spectrumslotnumber"]),
            demandvalues=tuple(int(v) for v in r["demandvalues"].split()),
        )
        out[(p.traffic_set, p.topology, p.pattern)] = p
    return out


def topologies(traffic_set: str | None = None, role: str | None = None) -> list[str]:
    """Released topology names, optionally restricted to a traffic set and a role.

    The roles are ``reference`` (16 Topology Zoo networks), ``deeprmsa``
    (NsfnetChen, Cost239) and ``large`` (Dfn, Renater2010; ``demands50`` only).
    """
    return sorted({p.topology for p in _table().values()
                   if traffic_set in (None, p.traffic_set) and role in (None, p.role)})


def get_parameters(traffic_set: str, topology: str, pattern: str) -> TrafficParameters:
    try:
        return _table()[(traffic_set, topology, pattern)]
    except KeyError:
        raise KeyError(
            f"no parameters for ({traffic_set!r}, {topology!r}, {pattern!r}); "
            f"traffic sets: {TRAFFIC_SETS}, topologies: {topologies()}"
        ) from None


@cache
def _graphml(topology: str) -> str:
    if topology not in topologies():
        raise KeyError(f"unknown topology {topology!r}; topologies: {topologies()}")
    return resources.files(__package__).joinpath(f"data/topologies/{topology}.graphml").read_text()


def load_topology(topology: str) -> nx.DiGraph:
    """Released graph of ``topology`` with node ids 0..N-1."""
    import io

    return nx.read_graphml(io.StringIO(_graphml(topology)), node_type=int)
