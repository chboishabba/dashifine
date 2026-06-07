"""
Emit canonical Dashifine conversational path rows and token stream for TextGraphs.

Artifacts:
- dashifine_path_rows.csv  (t,label,self,norm,mirror,R,z,phase)
- dashifine_token_stream.txt  (space-separated LABEL:self,norm,mirror tokens)
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conversational_path_plot import build_path  # noqa: E402

ROOT = Path(__file__).resolve().parent
ROWS_OUT = ROOT / "dashifine_path_rows.csv"
TOKENS_OUT = ROOT / "dashifine_token_stream.txt"


def resonance(x: int, y: int, z: int) -> int:
    return x * y + y * z + z * x


def label_phase(idx: int) -> str:
    # A–E are phase1, E1–I are phase2 in the archived script.
    return "phase1" if idx < 5 else "phase2"


def emit_rows_and_tokens():
    path = build_path()  # list of (label, self, norm, mirror)
    rows = []
    tokens = []
    for t, (lab, s, n, m) in enumerate(path, start=1):
        R = resonance(s, n, m)
        z = s * n
        rows.append(
            {
                "t": t,
                "label": lab,
                "self": s,
                "norm": n,
                "mirror": m,
                "R": R,
                "z": z,
                "phase": label_phase(t - 1),
            }
        )
        tokens.append(f"{lab}:{s:+d},{n:+d},{m:+d}")

    # CSV output
    with ROWS_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["t", "label", "self", "norm", "mirror", "R", "z", "phase"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # Token stream
    TOKENS_OUT.write_text(" ".join(tokens) + "\n", encoding="utf-8")

    return ROWS_OUT, TOKENS_OUT


def main():
    rows_path, tokens_path = emit_rows_and_tokens()
    print(f"wrote {rows_path}")
    print(f"wrote {tokens_path}")


if __name__ == "__main__":
    main()
