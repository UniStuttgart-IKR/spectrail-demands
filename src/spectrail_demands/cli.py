"""Command line: write generated sequences to CSV or DuckDB.

    spectrail-demands list
    spectrail-demands generate --traffic-set demands50 --topology Iinet \\
        --pattern constanterlang --seeds 1000001-1001000 --out iinet_ce.duckdb

DuckDB output uses the ``demand_sequences`` layout of the spectrail-rsa-episodes
``id0`` file and a ``demands`` table with 0-based node ids and ``idx`` from 1.
``traffic_set`` holds ``generated_`` plus the chosen traffic set, and
``sequence_id`` holds the seed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from spectrail_demands import PATTERNS, TRAFFIC_SETS, generate_sequence, get_parameters, topologies

_COLUMNS = ("traffic_set", "graph_id", "pattern", "sequence_id", "idx",
            "source", "target", "demandvalue", "tin", "tout")


def _seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def _rows(args, seed):
    demands = generate_sequence(args.topology, args.pattern, seed,
                                traffic_set=args.traffic_set, n=args.n)
    for i, d in enumerate(demands, 1):
        yield (f"generated_{args.traffic_set}", args.topology, args.pattern, seed, i,
               d.source, d.target, d.demandvalue, d.tin, d.tout)


def _write_csv(args, seeds):
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_COLUMNS)
        for s in seeds:
            w.writerows(_rows(args, s))


def _write_duckdb(args, seeds):
    try:
        import duckdb
    except ImportError:
        sys.exit("DuckDB output needs the 'duckdb' extra: pip install 'spectrail-demands[duckdb]'")
    p = get_parameters(args.traffic_set, args.topology, args.pattern)
    con = duckdb.connect(str(args.out))
    con.execute("""
        CREATE TABLE IF NOT EXISTS demand_sequences (
            traffic_set VARCHAR, graph_id VARCHAR, pattern VARCHAR, sequence_id INTEGER,
            demandnum INTEGER, interarrivaltime DOUBLE, servicetime DOUBLE,
            spectrumslotnumber INTEGER,
            PRIMARY KEY (traffic_set, graph_id, pattern, sequence_id));
        CREATE TABLE IF NOT EXISTS demands (
            traffic_set VARCHAR, graph_id VARCHAR, pattern VARCHAR, sequence_id INTEGER,
            idx INTEGER, source INTEGER, target INTEGER, demandvalue TINYINT,
            tin DOUBLE, tout DOUBLE);
    """)
    for s in seeds:
        rows = list(_rows(args, s))
        con.execute("INSERT INTO demand_sequences VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [f"generated_{args.traffic_set}", args.topology, args.pattern, s, len(rows),
                     p.interarrivaltime, p.servicetime, p.spectrumslotnumber])
        con.executemany("INSERT INTO demands VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spectrail-demands", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="released topologies and parameters")
    g = sub.add_parser("generate", help="write sequences")
    g.add_argument("--traffic-set", required=True, choices=TRAFFIC_SETS)
    g.add_argument("--topology", required=True)
    g.add_argument("--pattern", required=True, choices=PATTERNS)
    g.add_argument("--seeds", required=True, help="e.g. 1000001-1001000 or 5,7,9")
    g.add_argument("--n", type=int, default=None, help="demands per sequence (default: released demandnum)")
    g.add_argument("--out", required=True, type=Path, help=".csv or .duckdb")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        print("traffic_set\ttopology\tpattern\tdemandnum\tservicetime\tslots\twidths")
        for ts in TRAFFIC_SETS:
            for t in topologies(ts):
                for pat in PATTERNS:
                    p = get_parameters(ts, t, pat)
                    widths = " ".join(map(str, p.demandvalues))
                    print(f"{ts}\t{t}\t{pat}\t{p.demandnum}\t{p.servicetime:.4f}"
                          f"\t{p.spectrumslotnumber}\t{widths}")
        return

    seeds = _seeds(args.seeds)
    if args.out.suffix == ".csv":
        _write_csv(args, seeds)
    elif args.out.suffix == ".duckdb":
        _write_duckdb(args, seeds)
    else:
        sys.exit("--out must end in .csv or .duckdb")


if __name__ == "__main__":
    main()
