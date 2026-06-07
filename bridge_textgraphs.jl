# Run a TextGraphs-compatible bridge over the canonical Dashifine path outputs.
#
# Outputs:
#   textgraphs_graph_props.csv
#   textgraphs_weighted_summary.csv
#   textgraphs_window_summary.csv
#   textgraphs_weighted_edges.csv
#
# Usage:
#   env JULIA_DEPOT_PATH=/home/c/Documents/code/.julia_depot \
#     julia --startup-file=no --compiled-modules=no dashifine/bridge_textgraphs.jl

using Graphs
using MetaGraphs
using CSV
using DataFrames
using Statistics

const ROOT = @__DIR__
const TOKENS_PATH = joinpath(ROOT, "dashifine_token_stream.txt")
const ROWS_PATH = joinpath(ROOT, "dashifine_path_rows.csv")

const PROPS_OUT = joinpath(ROOT, "textgraphs_graph_props.csv")
const PROPS_AUG_OUT = joinpath(ROOT, "textgraphs_graph_props_augmented.csv")
const WEIGHTED_SUMMARY_OUT = joinpath(ROOT, "textgraphs_weighted_summary.csv")
const WINDOW_SUMMARY_OUT = joinpath(ROOT, "textgraphs_window_summary.csv")
const WEIGHTED_EDGES_OUT = joinpath(ROOT, "textgraphs_weighted_edges.csv")

function build_labelled_graph(tokens)
    unique_tokens = unique(tokens)
    g = DiGraph(length(unique_tokens))
    inds = indexin(tokens, unique_tokens)
    for i in 2:length(inds)
        add_edge!(g, inds[i - 1], inds[i])
    end
    mg = MetaDiGraph(g)
    for (token_index, unique_token) in enumerate(unique_tokens)
        set_prop!(mg, token_index, :token, unique_token)
    end
    return mg
end

function mean_or_missing(xs)
    isempty(xs) ? missing : mean(xs)
end

function graph_props(g)
    SCC = strongly_connected_components(g)
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
        "num_strong_connect_comp" => length(SCC),
        "size_largest_scc" => maximum(map(length, SCC)),
        "graph_size" => nv(g),
    )
end

load_rows() = CSV.read(ROWS_PATH, DataFrame)

function as_int(v)
    parse(Int, string(v))
end

function transition_weight(a, b)
    dR = abs(as_int(b.R) - as_int(a.R))
    dz = abs(as_int(b.z) - as_int(a.z))
    dstate = abs(as_int(b.self) - as_int(a.self)) +
             abs(as_int(b.norm) - as_int(a.norm)) +
             abs(as_int(b.mirror) - as_int(a.mirror))
    weight = 1 + dR + dz + dstate
    return weight, dR, dz, dstate
end

function window_slices(rows, size::Int=3)
    # return windows as DataFrame slices for simplicity
    windows = Vector{SubDataFrame}()
    for i in 1:(nrow(rows) - size + 1)
        push!(windows, view(rows, i:(i + size - 1), :))
    end
    return windows
end

function write_df(path, df::DataFrame)
    CSV.write(path, df)
end

function main()
    tokens_line = String(read(TOKENS_PATH))
    tokens = split(strip(tokens_line))
    g = build_labelled_graph(tokens)
    props = graph_props(g)
    props_df = DataFrame(key = collect(keys(props)), value = collect(values(props)))
    write_df(PROPS_OUT, props_df)

    rows = load_rows()

    # Augmented graph with similarity edges (ternary state distance <= 1)
    label_to_state = Dict{String,NTuple{3,Int}}()
    for r in eachrow(rows)
        label_to_state[string(r.label)] = (as_int(r.self), as_int(r.norm), as_int(r.mirror))
    end
    aug_g = DiGraph(nv(g))
    for e in edges(g)
        add_edge!(aug_g, src(e), dst(e))
    end
    uniq = unique(tokens)
    for i in 1:length(uniq)
        for j in 1:length(uniq)
            if i == j
                continue
            end
            si = label_to_state[uniq[i]]
            sj = label_to_state[uniq[j]]
            dist = abs(si[1] - sj[1]) + abs(si[2] - sj[2]) + abs(si[3] - sj[3])
            if dist <= 1
                add_edge!(aug_g, i, j)
            end
        end
    end
    props_aug = graph_props(aug_g)
    props_aug_df = DataFrame(key = collect(keys(props_aug)), value = collect(values(props_aug)))
    write_df(PROPS_AUG_OUT, props_aug_df)

    edge_rows = DataFrame(
        from = String[],
        to = String[],
        weight = Int[],
        dR = Int[],
        dz = Int[],
        dstate = Int[],
    )
    weights = Int[]
    for i in 1:(nrow(rows) - 1)
        weight, dR, dz, dstate = transition_weight(rows[i, :], rows[i + 1, :])
        push!(weights, weight)
        push!(edge_rows, (
            from = string(rows[i, :label]),
            to = string(rows[i + 1, :label]),
            weight = weight,
            dR = dR,
            dz = dz,
            dstate = dstate,
        ))
    end
    write_df(WEIGHTED_EDGES_OUT, edge_rows)

    weighted_summary = DataFrame(
        key = String[
            "transition_count",
            "weight_min",
            "weight_max",
            "weight_mean",
            "dR_sum",
            "dz_sum",
            "dstate_sum",
        ],
        value = Any[
            length(weights),
            isempty(weights) ? missing : minimum(weights),
            isempty(weights) ? missing : maximum(weights),
            isempty(weights) ? missing : mean(weights),
            nrow(edge_rows) == 0 ? 0 : sum(edge_rows.dR),
            nrow(edge_rows) == 0 ? 0 : sum(edge_rows.dz),
            nrow(edge_rows) == 0 ? 0 : sum(edge_rows.dstate),
        ],
    )
    write_df(WEIGHTED_SUMMARY_OUT, weighted_summary)

    window_rows = DataFrame(
        window_start = Int[],
        labels = String[],
        phase_mix = String[],
        R_mean = Float64[],
        R_range = Int[],
        z_mean = Float64[],
        z_range = Int[],
        transition_weight_sum = Int[],
        unique_labels = Int[],
        repeat_count = Int[],
    )

    if nrow(rows) >= 3
        for i in 1:(nrow(rows) - 3 + 1)
            win = view(rows, i:(i + 2), :)
            wtrans = Int[]
            for j in 1:(nrow(win) - 1)
                weight, _, _, _ = transition_weight(win[j, :], win[j + 1, :])
                push!(wtrans, weight)
            end
            if isempty(wtrans)
                continue
            end
            r_vals = [as_int(win[j, :].R) for j in 1:nrow(win)]
            z_vals = [as_int(win[j, :].z) for j in 1:nrow(win)]
            labels = [string(win[j, :label]) for j in 1:nrow(win)]
            phases = unique([string(win[j, :phase]) for j in 1:nrow(win)])
            counts = Dict{String,Int}()
            for lab in labels
                counts[lab] = get(counts, lab, 0) + 1
            end
            repeat_count = sum(max(v - 1, 0) for v in values(counts))
            push!(window_rows, (
                window_start = i,
                labels = join(labels, " "),
                phase_mix = join(sort(phases), " "),
                R_mean = mean(r_vals),
                R_range = maximum(r_vals) - minimum(r_vals),
                z_mean = mean(z_vals),
                z_range = maximum(z_vals) - minimum(z_vals),
                transition_weight_sum = isempty(wtrans) ? 0 : sum(wtrans),
                unique_labels = length(unique(labels)),
                repeat_count = repeat_count,
            ))
        end
    end
    write_df(WINDOW_SUMMARY_OUT, window_rows)

    println("wrote ", PROPS_OUT)
    println("wrote ", WEIGHTED_SUMMARY_OUT)
    println("wrote ", WINDOW_SUMMARY_OUT)
    println("wrote ", WEIGHTED_EDGES_OUT)
end

main()
