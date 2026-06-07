"""
Phase 2 bridge report for Dashifine conversational path.

Summarizes:
- weighted transition behavior from R/z
- sliding-window behavior over the A->I sequence

Outputs:
- bridge_phase2.json
- bridge_phase2.csv
"""

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROWS_PATH = ROOT / "dashifine_path_rows.csv"
JSON_OUT = ROOT / "bridge_phase2.json"
CSV_OUT = ROOT / "bridge_phase2.csv"


def load_rows():
    with ROWS_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_int(row, key):
    return int(row[key])


def window_slices(rows, size=3):
    for i in range(0, len(rows) - size + 1):
        yield i, rows[i : i + size]


def transition_weight(a, b):
    dR = abs(as_int(b, "R") - as_int(a, "R"))
    dz = abs(as_int(b, "z") - as_int(a, "z"))
    dstate = (
        abs(as_int(b, "self") - as_int(a, "self"))
        + abs(as_int(b, "norm") - as_int(a, "norm"))
        + abs(as_int(b, "mirror") - as_int(a, "mirror"))
    )
    return 1 + dR + dz + dstate, dR, dz, dstate


def main():
    rows = load_rows()

    transitions = []
    for a, b in zip(rows, rows[1:]):
        weight, dR, dz, dstate = transition_weight(a, b)
        transitions.append(
            {
                "from": a["label"],
                "to": b["label"],
                "weight": weight,
                "dR": dR,
                "dz": dz,
                "dstate": dstate,
            }
        )

    weighted_summary = {
        "transition_count": len(transitions),
        "weight_min": min(t["weight"] for t in transitions) if transitions else None,
        "weight_max": max(t["weight"] for t in transitions) if transitions else None,
        "weight_mean": sum(t["weight"] for t in transitions) / len(transitions) if transitions else None,
        "dR_sum": sum(t["dR"] for t in transitions),
        "dz_sum": sum(t["dz"] for t in transitions),
        "dstate_sum": sum(t["dstate"] for t in transitions),
    }

    window_rows = []
    for start, win in window_slices(rows, size=3):
        wtrans = [transition_weight(a, b)[0] for a, b in zip(win, win[1:])]
        r_vals = [as_int(r, "R") for r in win]
        z_vals = [as_int(r, "z") for r in win]
        labels = [r["label"] for r in win]
        window_rows.append(
            {
                "window_start": start + 1,
                "labels": " ".join(labels),
                "phase_mix": " ".join(sorted(set(r["phase"] for r in win))),
                "R_mean": sum(r_vals) / len(r_vals),
                "R_range": max(r_vals) - min(r_vals),
                "z_mean": sum(z_vals) / len(z_vals),
                "z_range": max(z_vals) - min(z_vals),
                "transition_weight_sum": sum(wtrans),
                "unique_labels": len(set(labels)),
                "repeat_count": sum(c - 1 for c in Counter(labels).values() if c > 1),
            }
        )

    summary = {
        "weighted": weighted_summary,
        "windows": window_rows,
        "bridge_read": {
            "note": "Weighted transitions make R/z visible to the bridge; window rows show local drift over the fixed conversational path.",
            "window_size": 3,
        },
    }

    with JSON_OUT.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in weighted_summary.items():
            writer.writerow([f"weighted_{key}", value])
        writer.writerow(["window_count", len(window_rows)])
        writer.writerow(["window_size", 3])
        writer.writerow(["window_with_max_transition_weight", max(window_rows, key=lambda r: r["transition_weight_sum"])["labels"] if window_rows else ""])

    print(f"wrote {JSON_OUT}")
    print(f"wrote {CSV_OUT}")


if __name__ == "__main__":
    main()
