import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, List
import textwrap, os, json

# ---------- Core executable spec ----------

TRIT_LABEL = {-1: "−", 0: "0", 1: "+"}

LENSES = [3, 6, 9]          # Self, Norm, Mirror
TIMES  = ["p", "0", "f"]    # past, now, future

LENS_NAME = {3: "Self", 6: "Norm", 9: "Mirror"}
TIME_NAME = {"p": "Past", "0": "Now", "f": "Future"}

@dataclass(frozen=True)
class Invariants:
    b0: Tuple[int, int, int]     # (S3,0, S6,0, S9,0)
    sigma3: int
    sigma6: int
    sigma9: int
    sigma: int
    delta_m: int
    tension: int
    neutral: int
    pos_count: int
    neg_count: int

@dataclass(frozen=True)
class Classification:
    motif: str   # "M1".."M10"
    action: str
    reason: str
    invariants: Invariants
    overflow: bool

def flips(trits: List[int]) -> int:
    # number of adjacent sign changes along time axis, including to/from neutral
    return int(trits[0] != trits[1]) + int(trits[1] != trits[2])

def default_mirror_expectation(S: np.ndarray, lens_row_map: Dict[int,int], t_col_map: Dict[str,int], t: str) -> int:
    """
    A reasonable *default* for an executable spec:
    Mirror is expected to be the involutive negation of Self at each time gate.
    expected_mirror(t) = - S_self(t)
    """
    self_val = int(S[lens_row_map[3], t_col_map[t]])
    return -self_val

def compute_invariants(
    S: np.ndarray,
    mirror_expected: Optional[Callable[[np.ndarray, Dict[int,int], Dict[str,int], str], int]] = None
) -> Invariants:
    """
    S: shape (3,3) with rows LENSES=[3,6,9] and cols TIMES=["p","0","f"] entries in {-1,0,1}
    mirror_expected: function giving expected Mirror value at time t, if you want Δm meaningful.
    """
    if mirror_expected is None:
        mirror_expected = default_mirror_expectation

    r = {lens:i for i,lens in enumerate(LENSES)}
    c = {t:j for j,t in enumerate(TIMES)}

    b0 = (int(S[r[3], c["0"]]), int(S[r[6], c["0"]]), int(S[r[9], c["0"]]))

    sigma3 = flips([int(S[r[3], c[t]]) for t in TIMES])
    sigma6 = flips([int(S[r[6], c[t]]) for t in TIMES])
    sigma9 = flips([int(S[r[9], c[t]]) for t in TIMES])
    sigma  = sigma3 + sigma6 + sigma9

    # Mirror asymmetry: per time gate, does S9,t match the expected mirror?
    delta_m = 0
    for t in TIMES:
        expected = int(mirror_expected(S, r, c, t))
        delta_m += int(int(S[r[9], c[t]]) != expected)

    flat = S.reshape(-1)
    pos_count = int(np.sum(flat == 1))
    neg_count = int(np.sum(flat == -1))
    neutral   = int(np.sum(flat == 0))
    tension   = pos_count * neg_count

    return Invariants(b0, sigma3, sigma6, sigma9, sigma, delta_m, tension, neutral, pos_count, neg_count)

def check_overflow(inv: Invariants, capacity: int = 9) -> bool:
    # The "lift trigger" as written: Z + T > 9 (you can swap this for a tighter condition later).
    return (inv.neutral + inv.tension) > capacity

def classify_motif(S: np.ndarray, inv: Optional[Invariants] = None) -> Classification:
    if inv is None:
        inv = compute_invariants(S)

    r = {lens:i for i,lens in enumerate(LENSES)}
    c = {t:j for j,t in enumerate(TIMES)}

    overflow = check_overflow(inv)

    # M10 is a resolution jump, not a normal motif.
    if overflow:
        return Classification(
            motif="M10",
            action="Lift / refine voxel (add new axis), then re-evaluate.",
            reason="Overflow trigger: (Z + T) > 9. Local compression unstable at current resolution.",
            invariants=inv,
            overflow=True
        )

    b0 = inv.b0
    S3_0, S6_0, S9_0 = b0

    # Priority (safety-first) classifier.
    if b0 == (-1, -1, -1):
        return Classification("M9", "Retire / prohibit; pivot.", "All-red present backbone b₀ = (−,−,−).", inv, overflow)

    if b0 == (1, 1, 1) and inv.sigma == 0 and inv.delta_m == 0:
        return Classification("M1", "Allow with standard guardrails.", "All-green backbone and stable (σ=0, Δm=0).", inv, overflow)

    if b0 == (1, 1, 1) and int(S[r[9], c["f"]]) == -1:
        return Classification("M2", "Allow with timing fences.", "Present is green but future Mirror is negative (S₉,f=−).", inv, overflow)

    if S3_0 == 1 and S9_0 == 1 and S6_0 == -1:
        return Classification("M3", "Role-gated: off-duty carve-outs / restrict in critical roles.", "Self+Mirror green but Norm red at present.", inv, overflow)

    if S3_0 == 1 and S6_0 == 1 and S9_0 == -1:
        return Classification("M5", "Buffer: lock context fences; do not generalize.", "Mirror flips negative at present despite Self+Norm green.", inv, overflow)

    if (S6_0 in (0,1)) and (S9_0 in (0,1)) and (S3_0 in (0,-1)):
        return Classification("M4", "Substrate redesign + re-trial.", "Externally coherent (Norm/Mirror ≥0) but Self not coherent (≤0).", inv, overflow)

    # Present harmful but Mirror shows an alternate constructive gate.
    if (S3_0 == -1 or S6_0 == -1 or S9_0 == -1) and (
        int(S[r[9], c["p"]]) == 1 or int(S[r[9], c["f"]]) == 1
    ):
        return Classification("M6", "Redesign lane: change timing/dose/product; reassess.", "Present harm but Mirror indicates a constructive path exists in past/future.", inv, overflow)

    # High temporal instability => fatigue / variance rim.
    if inv.sigma >= 3:
        return Classification("M7", "Time-domain control: taper/microdose/reset scheduling.", "High temporal flip count (σ large).", inv, overflow)

    # Sparse green occupancy => programmatic access only.
    if inv.pos_count <= 1 and inv.neg_count >= 5:
        return Classification("M8", "Programmatic access only; default to alternatives.", "Sparse positive occupancy; viability only in narrow windows.", inv, overflow)

    # Default conservative fallback: treat as M5 buffer.
    return Classification("M5", "Buffer: lock context fences; do not generalize.", "No decisive motif matched; conservative buffer default.", inv, overflow)


# ---------- Visualization (colored, technical + accessible) ----------

COLOR_MAP = {
    1:  "#2ecc71",  # green
    0:  "#bdc3c7",  # gray
    -1: "#e74c3c"   # red
}

def plot_state_with_report(S: np.ndarray, cls: Optional[Classification] = None, title: str = "Dashi state → invariants → motif"):
    if cls is None:
        cls = classify_motif(S)
    inv = cls.invariants

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Title
    ax.text(0.5, 0.96, title, ha="center", va="top", fontsize=14, fontweight="bold")

    # Draw grid
    x0, y0 = 0.07, 0.20
    cw, ch = 0.18, 0.18

    # Column labels
    for j, t in enumerate(TIMES):
        ax.text(x0 + j*cw + cw/2, y0 + 3*ch + 0.03, f"{TIME_NAME[t]} ({t})", ha="center", va="bottom", fontsize=11)

    # Row labels + cells
    for i, lens in enumerate(LENSES):
        ax.text(x0 - 0.02, y0 + (2-i)*ch + ch/2, f"{LENS_NAME[lens]} ({lens})", ha="right", va="center", fontsize=11)
        for j, t in enumerate(TIMES):
            v = int(S[i, j])
            rect = plt.Rectangle((x0 + j*cw, y0 + (2-i)*ch), cw, ch, facecolor=COLOR_MAP[v], edgecolor="black", linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x0 + j*cw + cw/2, y0 + (2-i)*ch + ch/2, TRIT_LABEL[v], ha="center", va="center", fontsize=18, fontweight="bold")

    # Legend
    ax.text(0.07, 0.14, "Legend:", fontsize=11, fontweight="bold")
    legend_items = [(1, "+ (constructive)"), (0, "0 (neutral/unknown)"), (-1, "− (harmful)")]
    lx = 0.16
    for k,(v,lab) in enumerate(legend_items):
        rect = plt.Rectangle((lx + 0.22*k, 0.125), 0.03, 0.03, facecolor=COLOR_MAP[v], edgecolor="black")
        ax.add_patch(rect)
        ax.text(lx + 0.22*k + 0.04, 0.14, lab, va="center", fontsize=10)

    # Technical report box (formulas + values)
    report = []
    report.append(f"Motif: {cls.motif}")
    report.append(f"Action: {cls.action}")
    report.append("")
    report.append("Invariants:")
    report.append(f"  b₀ = (S₃,0, S₆,0, S₉,0) = ({TRIT_LABEL[inv.b0[0]]}, {TRIT_LABEL[inv.b0[1]]}, {TRIT_LABEL[inv.b0[2]]})")
    report.append(f"  σ₃={inv.sigma3}, σ₆={inv.sigma6}, σ₉={inv.sigma9}  ⇒  σ={inv.sigma}")
    report.append(f"  Δₘ = {inv.delta_m}   (default mirror expectation: S₉,t ≈ −S₃,t)")
    report.append(f"  T = #(+)·#(−) = {inv.pos_count}·{inv.neg_count} = {inv.tension}")
    report.append(f"  Z = #(0) = {inv.neutral}")
    report.append(f"  Overflow: {cls.overflow}  (rule: Z + T > 9)")
    report.append("")
    report.append("Reason:")
    report.append(f"  {cls.reason}")
    report_text = "\n".join(report)

    ax.text(0.62, 0.78, "Executable spec report", fontsize=12, fontweight="bold", ha="left")
    ax.text(0.62, 0.75, report_text, fontsize=10, ha="left", va="top", family="monospace")

    return fig

# ---------- Demo outputs ----------

# A curated demo example (mostly "time-gated allow")
S_demo = np.array([
    [ 1,  1,  1],   # Self: past/now/future
    [ 1,  1,  1],   # Norm
    [ 1,  1, -1],   # Mirror flips negative in future
], dtype=int)

cls_demo = classify_motif(S_demo)
fig = plot_state_with_report(S_demo, cls_demo, title="Demo: 3×3 tensor → invariants → motif (colored)")
demo_path = "dashi_executable_demo.png"
fig.savefig(demo_path, dpi=200, bbox_inches="tight")
plt.close(fig)

# Save the spec as a runnable .py file
spec_path = "dashi_executable_spec.py"
spec_source = open(__file__, "r").read() if "__file__" in globals() else None

# We can't reliably read __file__ in this environment; write out a clean module instead.
module_text = f'''\
"""
Dashi (Dashifine) motifs M1..M10 — executable spec (Python)

- State tensor S ∈ T^(L×τ) where T={{-1,0,+1}}, L={{3,6,9}}, τ={{p,0,f}}
- Invariants: b0, sigma, delta_m, tension, neutral
- Deterministic classifier with priority order
- M10 is overflow/lift, not "one more motif"

This module is designed to be:
- readable for general audiences (colors + labels in plots)
- checkable for technical audiences (explicit invariants + formulas)
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt

TRIT_LABEL = {{-1: "−", 0: "0", 1: "+"}}

LENSES = [3, 6, 9]          # Self, Norm, Mirror
TIMES  = ["p", "0", "f"]    # past, now, future

LENS_NAME = {{3: "Self", 6: "Norm", 9: "Mirror"}}
TIME_NAME = {{"p": "Past", "0": "Now", "f": "Future"}}

@dataclass(frozen=True)
class Invariants:
    b0: Tuple[int, int, int]
    sigma3: int
    sigma6: int
    sigma9: int
    sigma: int
    delta_m: int
    tension: int
    neutral: int
    pos_count: int
    neg_count: int

@dataclass(frozen=True)
class Classification:
    motif: str
    action: str
    reason: str
    invariants: Invariants
    overflow: bool

def flips(trits: List[int]) -> int:
    return int(trits[0] != trits[1]) + int(trits[1] != trits[2])

def default_mirror_expectation(S: np.ndarray, lens_row_map: Dict[int,int], t_col_map: Dict[str,int], t: str) -> int:
    self_val = int(S[lens_row_map[3], t_col_map[t]])
    return -self_val

def compute_invariants(
    S: np.ndarray,
    mirror_expected: Optional[Callable[[np.ndarray, Dict[int,int], Dict[str,int], str], int]] = None
) -> Invariants:
    if mirror_expected is None:
        mirror_expected = default_mirror_expectation

    r = {{lens:i for i,lens in enumerate(LENSES)}}
    c = {{t:j for j,t in enumerate(TIMES)}}

    b0 = (int(S[r[3], c["0"]]), int(S[r[6], c["0"]]), int(S[r[9], c["0"]]))

    sigma3 = flips([int(S[r[3], c[t]]) for t in TIMES])
    sigma6 = flips([int(S[r[6], c[t]]) for t in TIMES])
    sigma9 = flips([int(S[r[9], c[t]]) for t in TIMES])
    sigma  = sigma3 + sigma6 + sigma9

    delta_m = 0
    for t in TIMES:
        expected = int(mirror_expected(S, r, c, t))
        delta_m += int(int(S[r[9], c[t]]) != expected)

    flat = S.reshape(-1)
    pos_count = int(np.sum(flat == 1))
    neg_count = int(np.sum(flat == -1))
    neutral   = int(np.sum(flat == 0))
    tension   = pos_count * neg_count

    return Invariants(b0, sigma3, sigma6, sigma9, sigma, delta_m, tension, neutral, pos_count, neg_count)

def check_overflow(inv: Invariants, capacity: int = 9) -> bool:
    return (inv.neutral + inv.tension) > capacity

def classify_motif(S: np.ndarray, inv: Optional[Invariants] = None) -> Classification:
    if inv is None:
        inv = compute_invariants(S)

    r = {{lens:i for i,lens in enumerate(LENSES)}}
    c = {{t:j for j,t in enumerate(TIMES)}}

    overflow = check_overflow(inv)
    if overflow:
        return Classification(
            motif="M10",
            action="Lift / refine voxel (add new axis), then re-evaluate.",
            reason="Overflow trigger: (Z + T) > 9. Local compression unstable at current resolution.",
            invariants=inv,
            overflow=True
        )

    b0 = inv.b0
    S3_0, S6_0, S9_0 = b0

    if b0 == (-1, -1, -1):
        return Classification("M9", "Retire / prohibit; pivot.", "All-red present backbone b₀ = (−,−,−).", inv, overflow)

    if b0 == (1, 1, 1) and inv.sigma == 0 and inv.delta_m == 0:
        return Classification("M1", "Allow with standard guardrails.", "All-green backbone and stable (σ=0, Δm=0).", inv, overflow)

    if b0 == (1, 1, 1) and int(S[r[9], c["f"]]) == -1:
        return Classification("M2", "Allow with timing fences.", "Present is green but future Mirror is negative (S₉,f=−).", inv, overflow)

    if S3_0 == 1 and S9_0 == 1 and S6_0 == -1:
        return Classification("M3", "Role-gated: off-duty carve-outs / restrict in critical roles.", "Self+Mirror green but Norm red at present.", inv, overflow)

    if S3_0 == 1 and S6_0 == 1 and S9_0 == -1:
        return Classification("M5", "Buffer: lock context fences; do not generalize.", "Mirror flips negative at present despite Self+Norm green.", inv, overflow)

    if (S6_0 in (0,1)) and (S9_0 in (0,1)) and (S3_0 in (0,-1)):
        return Classification("M4", "Substrate redesign + re-trial.", "Externally coherent (Norm/Mirror ≥0) but Self not coherent (≤0).", inv, overflow)

    if (S3_0 == -1 or S6_0 == -1 or S9_0 == -1) and (
        int(S[r[9], c["p"]]) == 1 or int(S[r[9], c["f"]]) == 1
    ):
        return Classification("M6", "Redesign lane: change timing/dose/product; reassess.", "Present harm but Mirror indicates a constructive path exists in past/future.", inv, overflow)

    if inv.sigma >= 3:
        return Classification("M7", "Time-domain control: taper/microdose/reset scheduling.", "High temporal flip count (σ large).", inv, overflow)

    if inv.pos_count <= 1 and inv.neg_count >= 5:
        return Classification("M8", "Programmatic access only; default to alternatives.", "Sparse positive occupancy; viability only in narrow windows.", inv, overflow)

    return Classification("M5", "Buffer: lock context fences; do not generalize.", "No decisive motif matched; conservative buffer default.", inv, overflow)

COLOR_MAP = {{
    1:  "#2ecc71",
    0:  "#bdc3c7",
    -1: "#e74c3c"
}}

def plot_state_with_report(S: np.ndarray, cls: Optional[Classification] = None, title: str = "Dashi state → invariants → motif"):
    if cls is None:
        cls = classify_motif(S)
    inv = cls.invariants

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.96, title, ha="center", va="top", fontsize=14, fontweight="bold")

    x0, y0 = 0.07, 0.20
    cw, ch = 0.18, 0.18

    for j, t in enumerate(TIMES):
        ax.text(x0 + j*cw + cw/2, y0 + 3*ch + 0.03, f"{{TIME_NAME[t]}} ({{t}})", ha="center", va="bottom", fontsize=11)

    for i, lens in enumerate(LENSES):
        ax.text(x0 - 0.02, y0 + (2-i)*ch + ch/2, f"{{LENS_NAME[lens]}} ({{lens}})", ha="right", va="center", fontsize=11)
        for j, t in enumerate(TIMES):
            v = int(S[i, j])
            rect = plt.Rectangle((x0 + j*cw, y0 + (2-i)*ch), cw, ch, facecolor=COLOR_MAP[v], edgecolor="black", linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x0 + j*cw + cw/2, y0 + (2-i)*ch + ch/2, TRIT_LABEL[v], ha="center", va="center", fontsize=18, fontweight="bold")

    report = []
    report.append(f"Motif: {{cls.motif}}")
    report.append(f"Action: {{cls.action}}")
    report.append("")
    report.append("Invariants:")
    report.append(f"  b₀ = (S₃,0, S₆,0, S₉,0) = ({{TRIT_LABEL[inv.b0[0]]}}, {{TRIT_LABEL[inv.b0[1]]}}, {{TRIT_LABEL[inv.b0[2]]}})")
    report.append(f"  σ₃={{inv.sigma3}}, σ₆={{inv.sigma6}}, σ₉={{inv.sigma9}}  ⇒  σ={{inv.sigma}}")
    report.append(f"  Δₘ = {{inv.delta_m}}   (default mirror expectation: S₉,t ≈ −S₃,t)")
    report.append(f"  T = #(+)·#(−) = {{inv.pos_count}}·{{inv.neg_count}} = {{inv.tension}}")
    report.append(f"  Z = #(0) = {{inv.neutral}}")
    report.append(f"  Overflow: {{cls.overflow}}  (rule: Z + T > 9)")
    report.append("")
    report.append("Reason:")
    report.append(f"  {{cls.reason}}")

    ax.text(0.62, 0.78, "Executable spec report", fontsize=12, fontweight="bold", ha="left")
    ax.text(0.62, 0.75, "\\n".join(report), fontsize=10, ha="left", va="top", family="monospace")

    return fig
"""
'''
with open(spec_path, "w") as f:
    f.write(module_text)

demo_path, spec_path
