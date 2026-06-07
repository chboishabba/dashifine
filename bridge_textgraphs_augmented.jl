# Build an augmented TextGraphs-compatible graph by adding non-local similarity
# edges between nearby ternary states, then emit graph properties.
#
# Outputs:
#   textgraphs_graph_props_augmented.csv
#
# Usage:
#   env JULIA_DEPOT_PATH=/home/c/Documents/code/.julia_depot \
#     julia --project=/home/c/Documents/code/TextGraphs.jl --startup-file=no --compiled-modules=no \
#       dashifine/bridge_textgraphs_augmented.jl

using Graphs
using CSV
using DataFrames
using Statistics

const ROOT = @__DIR__
const TOKENS_PATH = joinpath(ROOT, "dashifine_token_stream.txt")
const PROPS_AUG_OUT = joinpath(ROOT, "textgraphs_graph_props_augmented.csv")

function build_labelled_graph(tokens)
    unique_tokens = unique(tokens)
    g = DiGraph(length(unique_tokens))
    inds = indexin(tokens, unique_tokens)
    for i in 2:length(inds)
        add_edge!(g, inds[i - 1], inds[i])
    end
    return g, unique_tokens
end

function mean_or_missing(xs)
    isempty(xs) ? missing : mean(xs)
end

function graph_props(g)
    scc = strongly_connected_components(g)
    node_bet = betweenness_centrality(g)
    node_close = closeness_centrality(g)
    node_eig = try
        eigenvector_centrality(g)
    catch
        fill(missing, nv(g))
    end
    Dict(
        "mean_between_centr" => mean_or_missing(node_bet),
        "mean_close_centr" => mean_or_missing(node_close),
        "mean_eig_centr" => mean_or_missing(skipmissing(node_eig)),
        "density" => density(g),
        "num_self_loops" => num_self_loops(g),
        "num_strong_connect_comp" => length(scc),
        "size_largest_scc" => maximum(map(length, scc)),
        "graph_size" => nv(g),
    )
end

function as_int(v)
    parse(Int, string(v))
end

function main()
    tokens = split(strip(String(read(TOKENS_PATH))))
    g, uniq = build_labelled_graph(tokens)
    # Canonical token form is LABEL:+1,-1,0 etc. Extract state from token directly.
    function parse_state(tok::AbstractString)
        rhs = split(String(tok), ":")[2]
        vals = split(rhs, ",")
        return (parse(Int, vals[1]), parse(Int, vals[2]), parse(Int, vals[3]))
    end

    aug = DiGraph(nv(g))
    for e in edges(g)
        add_edge!(aug, src(e), dst(e))
    end

    for i in 1:length(uniq)
        for j in 1:length(uniq)
            if i == j
                continue
            end
            si = parse_state(uniq[i])
            sj = parse_state(uniq[j])
            dist = abs(si[1] - sj[1]) + abs(si[2] - sj[2]) + abs(si[3] - sj[3])
            if dist <= 1
                add_edge!(aug, i, j)
            end
        end
    end

    props = graph_props(aug)
    df = DataFrame(key = collect(keys(props)), value = collect(values(props)))
    CSV.write(PROPS_AUG_OUT, df)
    println("wrote ", PROPS_AUG_OUT)
end

main()
