import csv
import itertools
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from spectrail_demands import (
    PATTERNS,
    ROLES,
    TRAFFIC_SETS,
    demand_stream,
    generate_sequence,
    get_parameters,
    load_topology,
    mouseelephant_probabilities,
    to_arrays,
    topologies,
)
from spectrail_demands.derive import STDMULT, avamax_servicetime, steadystate_demandnumber

ALL = [(ts, t, p) for ts in TRAFFIC_SETS for t in topologies(ts) for p in PATTERNS]
REFERENCE = ["Basnet", "Epoch", "Gambia", "Gblnet", "Getnet", "Heanet", "HiberniaIreland", "Iinet",
             "Ilan", "Layer42", "Napnet", "Netrail", "Nordu1989", "Nordu2005", "Sanren", "Telecomserbia"]
WIDTHS = {"demands50": (2, 3, 4, 5), "demands100": (4, 6, 8)}
SLOTS = {"demands50": 50, "demands100": 100}


def test_release_coverage():
    assert topologies(role="reference") == REFERENCE
    assert topologies(role="deeprmsa") == ["Cost239", "NsfnetChen"]
    assert topologies(role="large") == ["Dfn", "Renater2010"]
    assert len(topologies()) == 20
    assert topologies("demands100") == sorted(REFERENCE + ["Cost239", "NsfnetChen"])
    assert len(ALL) == 114
    for ts, t, p in ALL:
        params = get_parameters(ts, t, p)
        assert params.role in ROLES
        assert params.demandvalues == WIDTHS[ts]
        assert params.spectrumslotnumber == SLOTS[ts]
    with pytest.raises(KeyError):
        load_topology("Mren")
    with pytest.raises(KeyError):
        get_parameters("demands100", "Dfn", "constanterlang")


@pytest.mark.parametrize("topology", topologies())
def test_graphml_node_ids_are_release_ids(topology):
    from spectrail_demands.parameters import _graphml

    ids = re.findall(r'<node id="([^"]+)"', _graphml(topology))
    assert ids == [str(i) for i in range(len(ids))]


@pytest.mark.parametrize("traffic_set,topology,pattern", ALL)
def test_servicetime_matches_release(traffic_set, topology, pattern):
    p = get_parameters(traffic_set, topology, pattern)
    G = load_topology(topology)
    st = avamax_servicetime(G, p.demandvalues, p.spectrumslotnumber, STDMULT[pattern])
    assert st == pytest.approx(p.servicetime, rel=1e-12)


@pytest.mark.parametrize("traffic_set", TRAFFIC_SETS)
@pytest.mark.parametrize("topology", ["Iinet", "Nordu1989", "Getnet"])
def test_demandnum_estimate_close_to_release(traffic_set, topology):
    p = get_parameters(traffic_set, topology, "constanterlang")
    n = steadystate_demandnumber(load_topology(topology), p.servicetime,
                                 p.demandvalues, p.spectrumslotnumber)
    assert abs(n - p.demandnum) / p.demandnum < 0.15


def test_mouseelephant_probabilities():
    probs = mouseelephant_probabilities([2, 3, 4, 5])
    pts = np.linspace(1, -np.log(0.001), 4)
    cdf = 1 - np.exp(-pts)
    np.testing.assert_allclose(probs, [cdf[0], cdf[1] - cdf[0], cdf[2] - cdf[1], 1 - cdf[2]])
    np.testing.assert_allclose(probs, [0.632, 0.317, 0.044, 0.007], atol=1e-3)
    np.testing.assert_allclose(mouseelephant_probabilities([4, 6, 8]), [0.632, 0.349, 0.019], atol=1e-3)


@pytest.mark.parametrize("traffic_set", TRAFFIC_SETS)
@pytest.mark.parametrize("pattern", PATTERNS)
def test_sequence_is_valid_and_deterministic(traffic_set, pattern):
    gen = lambda seed: generate_sequence("Iinet", pattern, seed, traffic_set=traffic_set)  # noqa: E731
    seq = gen(3)
    p = get_parameters(traffic_set, "Iinet", pattern)
    n_nodes = load_topology("Iinet").number_of_nodes()
    a = to_arrays(seq)
    assert len(seq) == p.demandnum
    assert np.all(a["source"] != a["target"])
    assert a["source"].min() >= 0 and max(a["source"].max(), a["target"].max()) < n_nodes
    assert set(a["demandvalue"]) <= set(p.demandvalues)
    assert np.all(np.diff(a["tin"]) > 0) and np.all(a["tout"] > a["tin"])
    assert seq == gen(3)
    assert seq != gen(4)
    stream = demand_stream("Iinet", pattern, 3, traffic_set=traffic_set)
    assert seq[:100] == list(itertools.islice(stream, 100))


DB = os.environ.get("SPECTRAIL_DATASET_DB")
NEEDS_DB = pytest.mark.skipif(
    DB is None, reason="set SPECTRAIL_DATASET_DB to the spectrail-rsa-episodes id0 file")
# The dataset has no episodes on the large topologies.
RELEASED_TOPOLOGIES = topologies(role="reference") + topologies(role="deeprmsa")


# Released demands with their index and time in the source sequence.
RELEASED = """
    SELECT d.sequence_id, d.source_idx AS idx, d.source, d.target, d.demandvalue,
           d.tin + e.source_time_offset AS tin, d.tout + e.source_time_offset AS tout
    FROM episode_demands d
    JOIN episodes e USING (traffic_set, graph_id, pattern, sequence_id, scope_start)
    WHERE d.traffic_set = '{ts}' AND d.graph_id = '{t}' AND d.pattern = '{p}'
      AND d.sequence_id IN ({seqs})
    ORDER BY d.sequence_id, d.source_idx"""


@NEEDS_DB
@pytest.mark.parametrize("topology", RELEASED_TOPOLOGIES)
def test_topology_matches_release(topology):
    import duckdb

    con = duckdb.connect(DB, read_only=True)
    edges = con.sql(f"SELECT source, target FROM edges WHERE graph_id = '{topology}'").fetchall()
    G = load_topology(topology)
    assert set(edges) == set(G.edges())
    assert sorted(G.nodes()) == list(range(G.number_of_nodes()))


@NEEDS_DB
@pytest.mark.parametrize("traffic_set,topology,pattern",
                         [c for c in ALL if c[1] in RELEASED_TOPOLOGIES])
def test_parameters_match_release(traffic_set, topology, pattern):
    import duckdb

    con = duckdb.connect(DB, read_only=True)
    rows = con.sql(f"""
        SELECT DISTINCT demandnum, interarrivaltime, servicetime, spectrumslotnumber
        FROM demand_sequences
        WHERE traffic_set = '{traffic_set}' AND graph_id = '{topology}' AND pattern = '{pattern}'
    """).fetchall()
    p = get_parameters(traffic_set, topology, pattern)
    assert rows == [(p.demandnum, p.interarrivaltime, p.servicetime, p.spectrumslotnumber)]


# About 60 two-sample tests run below; ALPHA keeps the family-wise false alarm
# rate near 1 %. Released episodes omit sequence tails and unproven solves, so
# they are a slightly selected sample of the generator output.
ALPHA = 1e-4


@NEEDS_DB
@pytest.mark.parametrize("traffic_set", TRAFFIC_SETS)
@pytest.mark.parametrize("topology", ["Iinet", "Telecomserbia"])
@pytest.mark.parametrize("pattern", PATTERNS)
def test_distributions_match_release(traffic_set, topology, pattern):
    """Generated and released demands pass two-sample tests on every marginal."""
    import duckdb
    from scipy import stats

    con = duckdb.connect(DB, read_only=True)
    seqs = ", ".join(map(str, range(102, 132)))
    db = con.sql(RELEASED.format(ts=traffic_set, t=topology, p=pattern, seqs=seqs)).fetchnumpy()
    gen = [to_arrays(generate_sequence(topology, pattern, s, traffic_set=traffic_set))
           for s in range(10**6, 10**6 + 30)]
    gen = {k: np.concatenate([g[k] for g in gen]) for k in gen[0]}
    # Inter-arrival times between consecutive demands of one source sequence.
    consecutive = (db["sequence_id"][1:] == db["sequence_id"][:-1]) & (np.diff(db["idx"]) == 1)
    db_iat = np.diff(db["tin"])[consecutive]
    gen_iat = np.diff(gen["tin"])
    gen_iat = gen_iat[gen_iat > 0]  # drop the jump at sequence boundaries
    assert stats.ks_2samp(db["tout"] - db["tin"], gen["tout"] - gen["tin"]).pvalue > ALPHA
    assert stats.ks_2samp(db_iat, gen_iat).pvalue > ALPHA
    for col in ("demandvalue", "source", "target"):
        k = max(db[col].max(), gen[col].max()) + 1
        table = np.vstack([np.bincount(db[col], minlength=k), np.bincount(gen[col], minlength=k)])
        table = table[:, table.sum(0) > 0]
        assert stats.chi2_contingency(table).pvalue > ALPHA, col


JULIA = os.environ.get("SPECTRAIL_TEST_JULIA") and shutil.which("julia")


@pytest.mark.skipif(DB is None or not JULIA,
                    reason="set SPECTRAIL_DATASET_DB and SPECTRAIL_TEST_JULIA=1 (instantiated julia/ project)")
@pytest.mark.parametrize("traffic_set", TRAFFIC_SETS)
@pytest.mark.parametrize("topology", ["Iinet", "Getnet", "NsfnetChen", "Cost239"])
def test_julia_reproduces_release_exactly(traffic_set, topology, tmp_path):
    import duckdb

    root = Path(__file__).resolve().parents[1]
    con = duckdb.connect(DB, read_only=True)
    role = get_parameters(traffic_set, topology, "constanterlang").role
    seqs = "1,2,101" if role == "deeprmsa" else "1,2,500"  # DeepRMSA releases sequences up to 101
    for pattern in PATTERNS:
        out = tmp_path / f"{pattern}.csv"
        subprocess.run(["julia", "--startup-file=no", f"--project={root / 'julia'}",
                        str(root / "julia" / "reproduce.jl"), traffic_set, topology, pattern,
                        seqs, str(out)],
                       check=True)
        gen = {(int(r["sequence_id"]), int(r["idx"])): r for r in csv.DictReader(out.open())}
        db = con.sql(RELEASED.format(ts=traffic_set, t=topology, p=pattern, seqs=seqs)).fetchall()
        assert db
        for seq, idx, source, target, width, tin, tout in db:
            r = gen[(seq, idx)]
            assert (int(r["source"]), int(r["target"]), int(r["demandvalue"])) == (source, target, width)
            # Episode times are shifted by source_time_offset, so the sum rounds.
            assert float(r["tin"]) == pytest.approx(tin, rel=1e-9, abs=1e-9)
            assert float(r["tout"]) == pytest.approx(tout, rel=1e-9, abs=1e-9)
