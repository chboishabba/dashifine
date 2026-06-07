"""
Compute simple correlations between Dashifine window metrics and TextGraphs window metrics.

Inputs:
- bridge_phase2.json (Dashifine window metrics)
- textgraphs_window_summary.csv (TextGraphs window metrics)

Outputs:
- bridge_correlation.csv
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PHASE2_JSON = ROOT / "bridge_phase2.json"
TG_WINDOW_CSV = ROOT / "textgraphs_window_summary.csv"
OUT_CSV = ROOT / "bridge_correlation.csv"


def main():
    with PHASE2_JSON.open() as f:
        phase2 = json.load(f)

    dash_win = pd.DataFrame(phase2["windows"])
    dash_win = dash_win.rename(columns={"window_start": "window_start_dash"})

    tg_win = pd.read_csv(TG_WINDOW_CSV)
    tg_win = tg_win.rename(columns={"window_start": "window_start_tg"})

    # Align by order (both are sequential windows of size 3)
    dash_win["idx"] = range(len(dash_win))
    tg_win["idx"] = range(len(tg_win))

    merged = pd.merge(dash_win, tg_win, on="idx", suffixes=("_dash", "_tg"))

    pairs = [
        ("R_mean_dash", "transition_weight_sum_tg"),
        ("R_range_dash", "transition_weight_sum_tg"),
        ("z_mean_dash", "transition_weight_sum_tg"),
        ("R_mean_dash", "R_mean_tg"),
        ("z_mean_dash", "z_mean_tg"),
    ]

    rows = []
    for a, b in pairs:
        if a in merged and b in merged:
            corr = merged[a].corr(merged[b])
            rows.append({"pair": f"{a} vs {b}", "pearson": corr})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
