"""Demand generators for the three traffic patterns.

Python port of the IKRNetBase.jl generators (``generateDemandSlotInOut``,
``generateDemandSlotInOut_MouseElephant``, ``generateDemandSlotInOutSinus``)
that produced the demand sequences of spectrail-rsa-episodes.

Every generator is an infinite stream; a finite sequence is the first
``demandnum`` demands of the stream. Each demand draws, in this order, the
source, the target, the width, the inter-arrival time and the holding time,
as in the Julia code.

The random streams differ from Julia's MersenneTwister, so a seed here does
not reproduce the sequence with the same ``sequence_id`` in the database. The
sampling distributions are identical.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

PATTERNS = ("constanterlang", "mouseelephant", "oscillatingarrival")

# Mouse-elephant: the largest width has probability ~BIG_VALUE_PROB.
BIG_VALUE_PROB = 0.001

# Oscillating arrivals (Spectrail.producedemands_sinus defaults).
SINUS_PERIOD_MULTIPLIER = 50.0
SINUS_PHASE = math.pi
SINUS_AMPLITUDE_MULTIPLIER = 0.9
SINUS_MIN_INTERARRIVAL = 0.01


@dataclass(frozen=True, slots=True)
class Demand:
    """One connection request. Node ids are 0-based."""

    source: int
    target: int
    demandvalue: int
    tin: float
    tout: float


def mouseelephant_probabilities(
    demandvalues: Sequence[int],
    bigvalprob: float = BIG_VALUE_PROB,
) -> np.ndarray:
    """Probabilities of the sorted unique widths under the mouse-elephant pattern.

    IKRNetBase ``getcategoricaldistr``: place ``len(values)`` points evenly on
    [1, -ln(bigvalprob)] on an Exp(1) axis. Bin i holds the Exp(1) mass between
    point i-1 and point i (point -1 = 0), and the last bin takes the remainder.
    For widths 2..5 this gives (0.632, 0.317, 0.044, 0.007), for widths 4, 6, 8
    (0.632, 0.349, 0.019).
    """
    n = len(sorted(set(demandvalues)))
    points = np.linspace(1.0, -math.log(bigvalprob), n)
    cdf = lambda x: 1.0 - math.exp(-x)  # noqa: E731
    lows = np.concatenate(([0.0], points))[: n - 1]
    probs = [cdf(hi) - cdf(lo) for lo, hi in zip(lows, points[: n - 1])]
    probs.append(1.0 - sum(probs))
    return np.asarray(probs)


def _endpoints(nodes: list[int], rng: np.random.Generator) -> tuple[int, int]:
    src = nodes[rng.integers(len(nodes))]
    others = [v for v in nodes if v != src]
    return src, others[rng.integers(len(others))]


def stream_constanterlang(
    nodes: Sequence[int],
    demandvalues: Sequence[int],
    interarrival: float,
    service: float,
    rng: np.random.Generator,
) -> Iterator[Demand]:
    """Poisson arrivals, exponential holding times, uniform widths."""
    nodes = sorted(nodes)
    values = list(demandvalues)
    t = 0.0
    while True:
        src, dst = _endpoints(nodes, rng)
        dv = values[rng.integers(len(values))]
        t += rng.exponential(interarrival)
        yield Demand(src, dst, dv, t, t + rng.exponential(service))


def stream_mouseelephant(
    nodes: Sequence[int],
    demandvalues: Sequence[int],
    interarrival: float,
    service: float,
    rng: np.random.Generator,
    bigvalprob: float = BIG_VALUE_PROB,
) -> Iterator[Demand]:
    """Poisson arrivals, exponential holding times, mostly small widths."""
    nodes = sorted(nodes)
    values = sorted(set(demandvalues))
    probs = mouseelephant_probabilities(values, bigvalprob)
    t = 0.0
    while True:
        src, dst = _endpoints(nodes, rng)
        dv = values[rng.choice(len(values), p=probs)]
        t += rng.exponential(interarrival)
        yield Demand(src, dst, dv, t, t + rng.exponential(service))


def stream_oscillatingarrival(
    nodes: Sequence[int],
    demandvalues: Sequence[int],
    interarrival: float,
    service: float,
    rng: np.random.Generator,
    period_multiplier: float = SINUS_PERIOD_MULTIPLIER,
    phase: float = SINUS_PHASE,
    amplitude_multiplier: float = SINUS_AMPLITUDE_MULTIPLIER,
) -> Iterator[Demand]:
    """Exponential inter-arrival times whose mean follows a sinusoid in time.

    After a demand arrives at t, the next inter-arrival time has mean
    ``interarrival + amplitude * sin(2 pi t / period + phase)``, floored at
    0.01. The first one has mean ``interarrival``.
    """
    nodes = sorted(nodes)
    values = list(demandvalues)
    period = period_multiplier * interarrival
    amplitude = amplitude_multiplier * interarrival
    omega = 2.0 * math.pi / period
    t = 0.0
    next_iat = rng.exponential(interarrival)
    while True:
        src, dst = _endpoints(nodes, rng)
        dv = values[rng.integers(len(values))]
        t += next_iat
        yield Demand(src, dst, dv, t, t + rng.exponential(service))
        mean = interarrival + amplitude * math.sin(t * omega + phase)
        next_iat = rng.exponential(mean if mean > 0 else SINUS_MIN_INTERARRIVAL)


def stream(
    pattern: str,
    nodes: Sequence[int],
    demandvalues: Sequence[int],
    interarrival: float,
    service: float,
    rng: np.random.Generator,
) -> Iterator[Demand]:
    """Infinite demand stream for ``pattern``."""
    if pattern == "constanterlang":
        return stream_constanterlang(nodes, demandvalues, interarrival, service, rng)
    if pattern == "mouseelephant":
        return stream_mouseelephant(nodes, demandvalues, interarrival, service, rng)
    if pattern == "oscillatingarrival":
        return stream_oscillatingarrival(nodes, demandvalues, interarrival, service, rng)
    raise ValueError(f"unknown pattern {pattern!r}; expected one of {PATTERNS}")


def take(demands: Iterator[Demand], n: int) -> list[Demand]:
    """First ``n`` demands of a stream."""
    return list(itertools.islice(demands, n))
