"""
Compare the Dashifine Phase 1 path CSV against the TextGraphs-style graph props.

Outputs:
- bridge_comparison.json
- bridge_comparison.csv
"""

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROWS_PATH = ROOT / "dashifine_path_rows.csv"
PROPS_PATH = ROOT / "textgraphs_graph_props.csv"
JSON_OUT = ROOT / "bridge_comparison.json"
CSV_OUT = ROOT / "bridge_comparison.csv"


def load_rows():
    with ROWS_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_props():
    with PROPS_PATH.open(newline="", encoding="utf-8") as f:
        return {row["key"]: row["value"] for row in csv.DictReader(f)}


def as_int(row, key):
    return int(row[key])


def as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def main():
    rows = load_rows()
    props = load_props()

    labels = [r["label"] for r in rows]
    phases = Counter(r["phase"] for r in rows)
    path_length = sum(
        abs(as_int(b, "self") - as_int(a, "self"))
        + abs(as_int(b, "norm") - as_int(a, "norm"))
        + abs(as_int(b, "mirror") - as_int(a, "mirror"))
        for a, b in zip(rows, rows[1:])
    )
    r_values = [as_int(r, "R") for r in rows]
    z_values = [as_int(r, "z") for r in rows]
    recurrent_labels = [lab for lab, count in Counter(labels).items() if count > 1]

    summary = {
        "dashifine": {
            "rows": len(rows),
            "labels": labels,
            "unique_labels": len(set(labels)),
            "phases": dict(phases),
            "path_l1_length": path_length,
            "R_min": min(r_values),
            "R_max": max(r_values),
            "R_mean": sum(r_values) / len(r_values),
            "z_min": min(z_values),
            "z_max": max(z_values),
            "z_mean": sum(z_values) / len(z_values),
            "recurrent_labels": recurrent_labels,
        },
        "textgraphs": {
            "graph_size": as_float(props.get("graph_size")),
            "density": as_float(props.get("density")),
            "num_self_loops": as_float(props.get("num_self_loops")),
            "num_strong_connect_comp": as_float(props.get("num_strong_connect_comp")),
            "size_largest_scc": as_float(props.get("size_largest_scc")),
            "mean_between_centr": as_float(props.get("mean_between_centr")),
            "mean_close_centr": as_float(props.get("mean_close_centr")),
            "mean_eig_centr": as_float(props.get("mean_eig_centr")),
        },
        "bridge_read": {
            "shared_sequence": True,
            "shared_canonical_token_form": "LABEL:self,norm,mirror",
            "comparison_note": "Dashifine preserves ternary trajectory invariants; TextGraphs preserves graph recurrence/density/SCC on the same ordered token stream.",
        },
    }

    with JSON_OUT.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    rows_out = [
        ("dashifine_rows", str(len(rows))),
        ("dashifine_unique_labels", str(len(set(labels)))),
        ("dashifine_path_l1_length", str(path_length)),
        ("dashifine_R_mean", str(summary["dashifine"]["R_mean"])),
        ("dashifine_z_mean", str(summary["dashifine"]["z_mean"])),
        ("textgraphs_graph_size", str(summary["textgraphs"]["graph_size"])),
        ("textgraphs_density", str(summary["textgraphs"]["density"])),
        ("textgraphs_num_scc", str(summary["textgraphs"]["num_strong_connect_comp"])),
        ("textgraphs_mean_between_centr", str(summary["textgraphs"]["mean_between_centr"])),
        ("textgraphs_mean_close_centr", str(summary["textgraphs"]["mean_close_centr"])),
    ]
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerows(rows_out)

    print(f"wrote {JSON_OUT}")
    print(f"wrote {CSV_OUT}")


if __name__ == "__main__":
    main()
