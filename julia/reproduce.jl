# Reproduce released demand sequences (spectrail-rsa-episodes, spectrail-ilp-rsa) bit for bit.
#
#   julia --project=julia julia/reproduce.jl TRAFFIC_SET TOPOLOGY PATTERN SEQUENCES OUT.csv [INDEX_BASE]
#   julia --project=julia julia/reproduce.jl demands100 Iinet constanterlang 1-50 iinet.csv
#
# Sequence k of a traffic set, topology and pattern is the IKRNetBase.jl
# generator with MersenneTwister(k) and the released servicetime, demandnum and
# widths (read from ../src/spectrail_demands/data/parameters.csv). Recomputing
# servicetime can differ in the last bit, which changes tout.
#
# Output columns follow the Python CLI. Node ids start at INDEX_BASE (default 0,
# as in the id0 release file; 1 for the id1 file).

using IKRNetBase, Graphs, Random

const DATA = joinpath(@__DIR__, "..", "src", "spectrail_demands", "data")
const BIGVALPROB = 0.001
const SINUS = (period = 50.0, amplitude = 0.9, phase = pi)

function parameters(traffic_set, topology, pattern)
    lines = readlines(joinpath(DATA, "parameters.csv"))
    header = split(lines[1], ",")
    col(r, name) = r[findfirst(==(name), header)]
    for l in lines[2:end]
        r = split(l, ",")
        if col(r, "traffic_set") == traffic_set && col(r, "topology") == topology && col(r, "pattern") == pattern
            return (demandvalues = parse.(Int, split(col(r, "demandvalues"))),
                    servicetime = parse(Float64, col(r, "servicetime")),
                    demandnum = parse(Int, col(r, "demandnum")),
                    interarrival = parse(Float64, col(r, "interarrivaltime")))
        end
    end
    error("no parameters for $traffic_set/$topology/$pattern")
end

"Released graph; GraphML node id i is vertex i + 1."
function loadgraph(topology)
    xml = read(joinpath(DATA, "topologies", "$topology.graphml"), String)
    ids = [m[1] for m in eachmatch(r"<node id=\"([^\"]+)\"", xml)]
    ids == string.(0:length(ids)-1) || error("$topology: GraphML node ids are not 0..N-1")
    g = SimpleDiGraph(length(ids))
    for m in eachmatch(r"<edge\b[^>]*>", xml)
        s = parse(Int, match(r"source=\"([^\"]+)\"", m.match)[1])
        t = parse(Int, match(r"target=\"([^\"]+)\"", m.match)[1])
        add_edge!(g, s + 1, t + 1)
    end
    return g
end

function sequence(g, pattern, p, k)
    rng = MersenneTwister(k)
    if pattern == "constanterlang"
        IKRNetBase.generateDemandSlotInOut(g, p.demandvalues, p.interarrival, p.servicetime, p.demandnum, rng)
    elseif pattern == "mouseelephant"
        IKRNetBase.generateDemandSlotInOut_MouseElephant(g, p.demandvalues, p.interarrival, p.servicetime,
                                                         p.demandnum, rng, BIGVALPROB)
    elseif pattern == "oscillatingarrival"
        IKRNetBase.generateDemandSlotInOutSinus(g, p.demandvalues, p.interarrival, p.servicetime,
                                                SINUS.period * p.interarrival, SINUS.amplitude * p.interarrival,
                                                SINUS.phase, p.demandnum, rng)
    else
        error("unknown pattern $pattern")
    end
end

function parseseqs(spec)
    out = Int[]
    for part in split(spec, ",")
        lo, hi = occursin("-", part) ? split(part, "-") : (part, part)
        append!(out, parse(Int, lo):parse(Int, hi))
    end
    return out
end

function main(args)
    length(args) in (5, 6) ||
        error("usage: reproduce.jl TRAFFIC_SET TOPOLOGY PATTERN SEQUENCES OUT.csv [INDEX_BASE]")
    traffic_set, topology, pattern, seqs, out = args[1:5]
    base = length(args) == 6 ? parse(Int, args[6]) : 0
    p = parameters(traffic_set, topology, pattern)
    g = loadgraph(topology)
    open(out, "w") do io
        println(io, "traffic_set,graph_id,pattern,sequence_id,idx,source,target,demandvalue,tin,tout")
        for k in parseseqs(seqs), (i, d) in enumerate(sequence(g, pattern, p, k))
            println(io, join((traffic_set, topology, pattern, k, i, d.source - 1 + base,
                              d.target - 1 + base, d.demandvalue, repr(d.tin), repr(d.tout)), ","))
        end
    end
end

main(ARGS)
