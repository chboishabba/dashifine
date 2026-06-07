"""
Lock the current bridge artifact set, compare baseline vs augmented graph props,
score a small variant sweep, and name the current best bridge rule.

Inputs:
- textgraphs_graph_props.csv
- textgraphs_graph_props_augmented.csv
- textgraphs_variant_props.csv
- bridge_correlation.csv

Outputs:
- bridge_reference_snapshot.json
- bridge_variant_report.json
- bridge_variant_report.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE_PROPS = ROOT / "textgraphs_graph_props.csv"
AUG_PROPS = ROOT / "textgraphs_graph_props_augmented.csv"
VARIANT_PROPS = ROOT / "textgraphs_variant_props.csv"
CORR_CSV = ROOT / "bridge_correlation.csv"
SNAPSHOT_JSON = ROOT / "bridge_reference_snapshot.json"
REPORT_JSON = ROOT / "bridge_variant_report.json"
REPORT_CSV = ROOT / "bridge_variant_report.csv"


def read_key_value_csv(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["key"]: float(row["value"]) for row in rows}


def read_variant_metrics(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            variant = row["variant"]
            out.setdefault(variant, {})
            value = float(row["value"])
            out[variant][row["metric"]] = value
            out[variant]["added_edges"] = float(row["added_edges"])
    return out


def read_correlations(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["pair"]: float(row["pearson"]) for row in csv.DictReader(f)}


def variant_score(metrics: dict[str, float], baseline: dict[str, float]) -> float:
    base_scc = baseline["num_strong_connect_comp"]
    scc_term = 0.0
    if base_scc > 1:
        scc_term = (base_scc - metrics["num_strong_connect_comp"]) / (base_scc - 1.0)
    density_lift = metrics["density"] - baseline["density"]
    close_lift = metrics["mean_close_centr"] - baseline["mean_close_centr"]
    eig_lift = metrics["mean_eig_centr"] - baseline["mean_eig_centr"]
    between_lift = metrics["mean_between_centr"] - baseline["mean_between_centr"]
    complexity_penalty = metrics["added_edges"] / max(metrics["graph_size"] * (metrics["graph_size"] - 1), 1.0)
    return (
        1.25 * scc_term
        + density_lift
        + close_lift
        + eig_lift
        + 0.5 * between_lift
        - 0.5 * complexity_penalty
    )


def delta_summary(baseline: dict[str, float], augmented: dict[str, float]) -> dict[str, float]:
    keys = [
        "density",
        "num_strong_connect_comp",
        "size_largest_scc",
        "mean_close_centr",
        "mean_between_centr",
        "mean_eig_centr",
    ]
    return {f"{key}_delta": augmented[key] - baseline[key] for key in keys}


def main() -> None:
    baseline = read_key_value_csv(BASE_PROPS)
    augmented = read_key_value_csv(AUG_PROPS)
    correlations = read_correlations(CORR_CSV)
    variants = read_variant_metrics(VARIANT_PROPS)

    snapshot = {
        "baseline_props": baseline,
        "augmented_props": augmented,
        "correlations": correlations,
        "variants_available": sorted(variants.keys()),
    }
    with SNAPSHOT_JSON.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)

    scored = []
    for name, metrics in variants.items():
        scored.append(
            {
                "variant": name,
                "score": variant_score(metrics, baseline),
                "added_edges": metrics["added_edges"],
                "density": metrics["density"],
                "num_strong_connect_comp": metrics["num_strong_connect_comp"],
                "size_largest_scc": metrics["size_largest_scc"],
                "mean_close_centr": metrics["mean_close_centr"],
                "mean_between_centr": metrics["mean_between_centr"],
                "mean_eig_centr": metrics["mean_eig_centr"],
            }
        )
    scored.sort(key=lambda row: row["score"], reverse=True)
    best = scored[0]

    report = {
        "baseline_vs_augmented": {
            "baseline": baseline,
            "augmented": augmented,
            "delta": delta_summary(baseline, augmented),
        },
        "correlation_read": correlations,
        "variant_ranking": scored,
        "selected_rule": {
            "variant": best["variant"],
            "score": best["score"],
            "selection_basis": (
                "Maximize SCC collapse and centrality/density lift from baseline "
                "while penalizing extra edges."
            ),
        },
    }

    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "variant",
                "score",
                "added_edges",
                "density",
                "num_strong_connect_comp",
                "size_largest_scc",
                "mean_close_centr",
                "mean_between_centr",
                "mean_eig_centr",
            ]
        )
        for row in scored:
            writer.writerow(
                [
                    row["variant"],
                    row["score"],
                    row["added_edges"],
                    row["density"],
                    row["num_strong_connect_comp"],
                    row["size_largest_scc"],
                    row["mean_close_centr"],
                    row["mean_between_centr"],
                    row["mean_eig_centr"],
                ]
            )

    print(f"wrote {SNAPSHOT_JSON}")
    print(f"wrote {REPORT_JSON}")
    print(f"wrote {REPORT_CSV}")


if __name__ == "__main__":
    main()
