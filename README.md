# spectrail-demands

This package generates the demand sequences of the spectrail-rsa-episodes
dataset. It draws dynamic routing and spectrum assignment (RSA) traffic from
the same distributions and on the same topologies as the two traffic sets of
the dataset, `demands50` and `demands100`. Use it to train a learning agent
on fresh sequences and to compare it with the ILP and heuristic decisions of
the released episodes.

The dataset landing page is
<https://content.ikr.uni-stuttgart.de/Content/RSA-ILP-dataset/>.

## Installation

```bash
pip install git+https://github.com/UniStuttgart-IKR/spectrail-demands.git
pip install "spectrail-demands[duckdb] @ git+https://github.com/UniStuttgart-IKR/spectrail-demands.git"  # DuckDB output
```

The package needs Python 3.10 or later, NumPy and NetworkX.

## Usage

Every call selects a traffic set, because the two sets differ in capacity and
demand widths.

```python
from spectrail_demands import generate_sequence, demand_stream, load_topology

G = load_topology("Iinet")  # directed graph, nodes 0..N-1
seq = generate_sequence("Iinet", "constanterlang", seed=1_000_001, traffic_set="demands50")
d = seq[0]  # Demand(source, target, demandvalue, tin, tout)

for d in demand_stream("Iinet", "oscillatingarrival", seed=7, traffic_set="demands100"):  # infinite
    ...
```

`generate_sequence` returns as many demands as a released sequence of the
same traffic set, topology and pattern (`demandnum`). `demand_stream`
continues without limit. Use it to train without episode boundaries.

The command line writes sequences as CSV or DuckDB:

```bash
spectrail-demands list
spectrail-demands generate --traffic-set demands50 --topology Iinet \
    --pattern constanterlang --seeds 1000001-1001000 --out iinet_constanterlang.duckdb
```

DuckDB output creates the tables `demand_sequences` and `demands`.
`demand_sequences` has the columns of the dataset table. `demands` holds one
row per demand with 0-based node ids and `idx` from 1. The column
`traffic_set` holds `generated_demands50` or `generated_demands100`, and
`sequence_id` holds the seed. A `.csv` path writes the `demands` columns only.

## Topologies

The package covers 20 topologies in three roles. `topologies(role=...)` and
`topologies(traffic_set)` select them.

| Role | Topologies | Traffic sets | Sequences in spectrail-rsa-episodes v2.3.0 |
|:-|:-|:-|:-|
| `reference` | Basnet, Epoch, Gambia, Gblnet, Getnet, Heanet, HiberniaIreland, Iinet, Ilan, Layer42, Napnet, Netrail, Nordu1989, Nordu2005, Sanren, Telecomserbia | both | 1 to 1000 |
| `deeprmsa` | NsfnetChen, Cost239 | both | 1 to 50, 52 to 101 |
| `large` | Dfn, Renater2010 | `demands50` | none |

The dataset contains no episodes on the large topologies. The package
provides their demands only, with the parameters of the original generator.

The reference and large topologies come from the Internet Topology Zoo. The
DeepRMSA topologies are the 14-node NSFNET and the 11-node COST239 networks
of DeepRMSA. Their GraphML files store the link length in km as `length_km`
and as `weight`, and the released candidate paths of these two topologies
are ranked by length. Demand generation uses hop counts on every topology.

Getnet uses the symmetric 14-edge graph of spectrail-rsa-episodes v2.2.0
and later. Releases up to v2.1.1 used a 12-edge graph with two unpaired
edges, so their Getnet sequences differ from the ones this package
describes.

## Traffic model

A demand asks for a contiguous block of slots between a source and a
different target node. It holds the block from its arrival time `tin` to its
departure time `tout`. The source and the target follow a uniform
distribution over the nodes. Inter-arrival times have mean 1. Holding times
follow an exponential distribution with the mean `servicetime` of the traffic
set, topology and pattern.

| Traffic set | Slots per edge | Widths (slots) | Mouse-elephant width probabilities |
|:-|-:|:-|:-|
| `demands50` | 50 | 2, 3, 4, 5 | 0.632, 0.317, 0.044, 0.007 |
| `demands100` | 100 | 4, 6, 8 | 0.632, 0.349, 0.019 |

| Pattern | Arrivals | Widths |
|:-|:-|:-|
| `constanterlang` | Poisson process | uniform |
| `mouseelephant` | Poisson process | mostly small, probabilities above |
| `oscillatingarrival` | exponential inter-arrival time whose mean follows `1 + 0.9 sin(2 pi t / 50 + pi)` | uniform |

IKRNetBase.jl sets the mean holding time so that the offered load approaches
full occupancy of the spectrum. The mean holding time equals
`C E / (h - s sh) / (w - s sw)` for a topology with `E` directed edges and
`C` slots per edge. Here `h` and `sh` are the mean and the sample standard
deviation of the hop count over all node pairs. `w` and `sw` are the mean and
the sample standard deviation of the widths. The factor `s` is 1 for
`constanterlang` and `mouseelephant` and 0.7 for `oscillatingarrival`.

## Released parameters

`src/spectrail_demands/data/parameters.csv` lists the role, `demandnum`,
`servicetime`, `interarrivaltime`, `spectrumslotnumber` and the widths of
each traffic set, topology and pattern, 114 rows in total. For the
reference and DeepRMSA topologies, the values come from the
`demand_sequences` table of spectrail-rsa-episodes v2.3.0, and
`src/spectrail_demands/data/topologies/` holds each graph as GraphML, copied
from the `graphs` table of its `id0` file. The GraphML node ids are the node
ids of that file (0 to N-1), so generated demands and released demands refer
to the same nodes without renumbering. Use the `paths` table of the dataset
as the candidate paths. The parameters of the large topologies come from
the IKRNetBase.jl generator, and `spectrail_demands.derive` reproduces their
`servicetime`.

`spectrail_demands.derive` computes `servicetime` and `demandnum` for a graph
outside the table. It reproduces every released `servicetime`. Its
`demandnum` estimate differs from the released value by a few percent,
because the count comes from a seeded simulation.

## Relation to the released sequences

A seed reproduces its sequence on every platform with the same NumPy version
(PCG64 generator). The generated and released sequences follow the same
distributions. The test suite checks this with two-sample tests on the
inter-arrival times, holding times, widths, sources and targets for two
topologies, both traffic sets and all patterns.

Generated sequences are independent draws from the released distributions.
Seeds of one million and above stay clear of the released sequence numbers.

## Exact reproduction

IKRNetBase.jl produced every released sequence. Sequence `k` of a traffic
set, topology and pattern uses `MersenneTwister(k)`. The Python package does
not reproduce these sequences. `julia/reproduce.jl` does, bit for bit, with
IKRNetBase.jl, Graphs.jl and Distributions.jl pinned in
`julia/Manifest.toml`:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia julia/reproduce.jl demands50 Iinet constanterlang 1-50 iinet.csv
```

The script reads `servicetime`, `demandnum` and the widths from the parameter
table. Recomputing `servicetime` in Julia 1.12 can change its last bit, and
with it every departure time. The output uses the columns of the command line
and 0-based node ids. A sixth argument `1` writes 1-based node ids.

The released episodes keep the index of each demand in its source
sequence (`episode_demands.source_idx`). Adding `episodes.source_time_offset`
to `tin` and `tout` restores the source times up to floating-point rounding.

## Tests

```bash
pip install -e ".[test]"
pytest
SPECTRAIL_DATASET_DB=/path/to/spectrail-rsa-episodes-v2.3.0-id0.duckdb pytest  # adds the comparison with the release
SPECTRAIL_TEST_JULIA=1 SPECTRAIL_DATASET_DB=... pytest  # adds the Julia reproduction (instantiated julia/ project)
```

The comparison with the release covers the reference and DeepRMSA
topologies.

## Provenance and license

The Python generators port `generateDemandSlotInOut`,
`generateDemandSlotInOut_MouseElephant`, `generateDemandSlotInOutSinus`,
`get_avamax_servicetime` and `get_almost_steadystate_demandnumber` of
IKRNetBase.jl (MIT license, copyright 2024 Filippos Christou and
contributors). The reference and large topologies derive from the Internet
Topology Zoo (S. Knight et al., "The Internet Topology Zoo", IEEE JSAC 29(9),
2011). The preprocessing removes nodes without coordinates, merges colocated
nodes and represents every link as two directed edges. The DeepRMSA
topologies follow the link lists of X. Chen et al., "DeepRMSA: A Deep
Reinforcement Learning Framework for Routing, Modulation and Spectrum
Assignment in Elastic Optical Networks", J. Lightw. Technol. 37(16), 2019.
The DeepRMSA link lists do not name the nodes, so the city names and
coordinates in these GraphML files are an assignment fitted to the link
lengths.

The MIT license covers the code (see `LICENSE`). The data files under
`src/spectrail_demands/data/`, the parameter table and the GraphML
topologies, are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), like the dataset
(see `LICENSE-DATA`).
