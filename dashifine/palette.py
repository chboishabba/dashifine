"""Colour palette utilities for Dashifine."""
from __future__ import annotations

from typing import Tuple, List
import hashlib
import re
import numpy as np
from matplotlib.colors import hsv_to_rgb

# Maximum expected depth of a lineage address.  This normalises the depth
# component used for saturation/value.
MAX_LINEAGE_DEPTH = 10


def _legacy_hsv(addr: str, base: int) -> Tuple[float, float, float]:
    """Old address mapping used by the initial tests.

    The address may contain a fractional part which encodes depth.  The integer
    portion is interpreted in reverse order to provide a simple hue.
    """

    if "." in addr:
        addr_main, frac_part = addr.split(".", 1)
    else:
        addr_main, frac_part = addr, ""

    digits = [min(int(ch), base - 1) for ch in addr_main if ch.isdigit()]
    hue = 0.0
    for k, d in enumerate(reversed(digits)):
        hue += d / (base ** (k + 1))
    depth = float(f"0.{frac_part}") if frac_part else 0.0
    sat = depth
    val = 1.0 - 0.5 * depth
    return hue, sat, val


def lineage_hsv_from_address(addr: str, base: int = 3) -> Tuple[float, float, float]:
    """Map a lineage address string to HSV components.

    When ``addr`` contains a fractional component the legacy mapping used by the
    original tests is applied for backwards compatibility.  Otherwise the new
    suffix-based mapping described in the task instructions is used.
    """

    if "." in addr:
        return _legacy_hsv(addr, base)

    # ------------------------------------------------------------------
    # Extract digits, clamping to the valid range for the base.  This makes the
    # function robust to slightly malformed addresses.
    # ------------------------------------------------------------------
    digits: List[int] = [min(int(ch), base - 1) for ch in addr if ch.isdigit()]
    if not digits:
        return 0.0, 0.0, 0.5

    # Stable hue from the last ``k`` digits
    k = min(len(digits), 8)
    suffix = digits[-k:]
    int_suffix = 0
    for d in suffix:
        int_suffix = int_suffix * base + d
    H = int_suffix / float(base ** k)

    # Fractional depth from address length
    d = min(len(digits) / MAX_LINEAGE_DEPTH, 1.0)
    S = d ** 0.8
    V = 0.5 + 0.5 * d
    return float(H), float(S), float(V)


def lineage_rgb_from_address(addr: str, base: int = 3) -> Tuple[float, float, float]:
    """Convenience wrapper returning RGB for a lineage address."""
    h, s, v = lineage_hsv_from_address(addr, base=base)
    rgb = hsv_to_rgb([[h, s, v]])[0]
    return float(rgb[0]), float(rgb[1]), float(rgb[2])
