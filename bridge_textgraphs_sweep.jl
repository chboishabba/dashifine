# Sweep a small fixed family of non-local edge rules over the canonical
# Dashifine token stream, then emit graph-property summaries for comparison.
#
# Outputs:
#   textgraphs_variant_props.csv
#
# Usage:
#   env JULIA_DEPOT_PATH=/home/c/Documents/code/.julia_depot \
#     julia --project=/home/c/Documents/code/TextGraphs.jl --startup-file=no --compiled-modules=no \
#       dashifine/bridge_textgraphs_sweep.jl

using Graphs
using CSV
using DataFrames
using Statistics

const ROOT = @__DIR__
const TOKENS_PATH = joinpath(ROOT, "dashifine_token_stream.txt")
const ROWS_PATH = joinpath(ROOT, "dashifine_path_rows.csv")
const OUT_PATH = joinpath(ROOT, "textgraphs_variant_props.csv")

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
        "edge_count" => ne(g),
    )
end

function parse_state(tok::AbstractString)
    rhs = split(String(tok), ":")[2]
    vals = split(rhs, ",")
    return (parse(Int, vals[1]), parse(Int, vals[2]), parse(Int, vals[3]))
end

function state_l1(a::NTuple{3,Int}, b::NTuple{3,Int})
    abs(a[1] - b[1]) + abs(a[2] - b[2]) + abs(a[3] - b[3])
end

function copy_graph(g)
    h = DiGraph(nv(g))
    for e in edges(g)
        add_edge!(h, src(e), dst(e))
    end
    return h
end

function add_state_threshold_edges!(g, states, threshold::Int)
    added = 0
    for i in 1:length(states)
        for j in 1:length(states)
            if i == j
                continue
            end
            if state_l1(states[i], states[j]) <= threshold && !has_edge(g, i, j)
                add_edge!(g, i, j)
                added += 1
            end
        end
    end
    return added
end

function add_recurrence_edges!(g, states)
    added = 0
    for i in 1:length(states)
        for j in 1:length(states)
            if i == j
                continue
            end
            if states[i] == states[j] && !has_edge(g, i, j)
                add_edge!(g, i, j)
                added += 1
            end
        end
    end
    return added
end

function add_rz_similarity_edges!(g, labels, rz_map; threshold::Int=1)
    added = 0
    for i in 1:length(labels)
        for j in 1:length(labels)
            if i == j
                continue
            end
            ai = rz_map[String(labels[i])]
            aj = rz_map[String(labels[j])]
            dist = abs(ai[1] - aj[1]) + abs(ai[2] - aj[2])
            if dist <= threshold && !has_edge(g, i, j)
                add_edge!(g, i, j)
                added += 1
            end
        end
    end
    return added
end

function emit_variant_rows!(rows, variant::String, g, added_edges::Int)
    props = graph_props(g)
    for (key, value) in props
        push!(rows, (variant = variant, metric = String(key), value = value, added_edges = added_edges))
    end
end

function main()
    tokens = split(strip(String(read(TOKENS_PATH))))
    base_g, uniq = build_labelled_graph(tokens)
    states = [parse_state(tok) for tok in uniq]

    rows_df = CSV.read(ROWS_PATH, DataFrame)
    rz_map = Dict{String,Tuple{Int,Int}}()
    for r in eachrow(rows_df)
        token = string(r.label, ":", r.self >= 0 ? "+" : "", r.self, ",",
                       r.norm >= 0 ? "+" : "", r.norm, ",",
                       r.mirror >= 0 ? "+" : "", r.mirror)
        rz_map[token] = (parse(Int, string(r.R)), parse(Int, string(r.z)))
    end

    out_rows = DataFrame(
        variant = String[],
        metric = String[],
        value = Any[],
        added_edges = Int[],
    )

    emit_variant_rows!(out_rows, "baseline", base_g, 0)

    g_l1_1 = copy_graph(base_g)
    added_l1_1 = add_state_threshold_edges!(g_l1_1, states, 1)
    emit_variant_rows!(out_rows, "ternary_l1_le_1", g_l1_1, added_l1_1)

    g_l1_2 = copy_graph(base_g)
    added_l1_2 = add_state_threshold_edges!(g_l1_2, states, 2)
    emit_variant_rows!(out_rows, "ternary_l1_le_2", g_l1_2, added_l1_2)

    g_recur = copy_graph(base_g)
    added_recur = add_recurrence_edges!(g_recur, states)
    emit_variant_rows!(out_rows, "state_recurrence", g_recur, added_recur)

    g_rz = copy_graph(base_g)
    added_rz = add_rz_similarity_edges!(g_rz, uniq, rz_map; threshold=1)
    emit_variant_rows!(out_rows, "rz_l1_le_1", g_rz, added_rz)

    g_hybrid = copy_graph(base_g)
    added_hybrid = add_state_threshold_edges!(g_hybrid, states, 1)
    added_hybrid += add_recurrence_edges!(g_hybrid, states)
    emit_variant_rows!(out_rows, "hybrid_l1_recurrence", g_hybrid, added_hybrid)

    CSV.write(OUT_PATH, out_rows)
    println("wrote ", OUT_PATH)
end

main()
