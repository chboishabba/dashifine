"""Colour palette utilities for Dashifine."""
from __future__ import annotations

from typing import Tuple, List
import numpy as np
from matplotlib.colors import hsv_to_rgb

# Maximum expected depth of a lineage address.  This normalises the depth
# component used for saturation/value.
MAX_LINEAGE_DEPTH = 10


def _validate_base(base: int) -> int:
    """Return ``base`` if it is a valid integer radix.

    The helper accepts regular Python integers as well as :class:`numpy`
    integer scalars while rejecting booleans (which are ``int`` subclasses).
    A ``ValueError`` is raised when the radix is smaller than two – the
    minimum meaningful base for positional representations.  Normalising the
    validation in one place keeps ``lineage_hue_from_address`` and the legacy
    helper in sync and avoids surprising ``ZeroDivisionError`` exceptions when
    callers accidentally pass ``base=0`` or ``base=1``.
    """

    if isinstance(base, bool) or not isinstance(base, (int, np.integer)):
        raise TypeError("base must be an integer")
    if base < 2:
        raise ValueError("base must be >= 2")
    return int(base)


def _legacy_hsv(addr: str, base: int) -> Tuple[float, float, float]:
    """Old address mapping used by the initial tests.

    The address may contain a fractional part which encodes depth.  The integer
    portion is interpreted in reverse order to provide a simple hue.
    """

    base = _validate_base(base)
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


def lineage_hue_from_address(addr: str, base: int = 3) -> Tuple[float, float, float]:
    """Map a lineage address string to placeholder HSV components.

    The mapping is deterministic and provides a stable hue so that repeated
    runs colour the same lineage consistently.  An optional fractional component
    encodes *depth* and modulates saturation and value.  This implementation is
    intentionally lightweight and does not attempt to match any formal p-adic
    specification; it simply offers a stable colouring hook.
    """

    base = _validate_base(base)

    if "." in addr:
        # Backwards compatible path matching the behaviour expected by the
        # original unit tests.  The integer portion of the address encodes hue
        # in reverse order and the fractional part determines saturation/value.
        return _legacy_hsv(addr, base)

    # Extract digits clamped to ``base`` to tolerate malformed input.
    digits: List[int] = [min(int(ch), base - 1) for ch in addr if ch.isdigit()]
    if not digits:
        # Default to mid-level grey if no digits are present.
        return 0.0, 0.0, 0.5

    # Stable hue from the last ``k`` digits interpreted as a base-``p`` suffix.
    k = min(len(digits), 8)
    suffix = digits[-k:]
    int_suffix = 0
    for d in suffix:
        int_suffix = int_suffix * base + d
    hue = int_suffix / float(base ** k)

    # Saturation/value from depth measured by address length.
    depth = min(len(digits) / MAX_LINEAGE_DEPTH, 1.0)
    sat = depth ** 0.8
    val = 0.5 + 0.5 * depth
    return float(hue), float(sat), float(val)


# Backwards compatibility ----------------------------------------------------
def lineage_hsv_from_address(addr: str, base: int = 3) -> Tuple[float, float, float]:
    """Backward compatible wrapper for :func:`lineage_hue_from_address`."""
    return lineage_hue_from_address(addr, base=base)


def lineage_rgb_from_address(addr: str, base: int = 3) -> Tuple[float, float, float]:
    """Convenience wrapper returning RGB for a lineage address."""
    h, s, v = lineage_hue_from_address(addr, base=base)
    rgb = hsv_to_rgb([[h, s, v]])[0]
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


def eigen_palette(W: np.ndarray) -> np.ndarray:
    """Temporary grayscale mapping for class weights.

    Until a proper PCA-based colouring is implemented the *eigen* palette simply
    collapses the class weight vectors to their mean and repeats the resulting
    grayscale value across the three RGB channels.
    """

    if W.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    gray = np.mean(W, axis=-1, keepdims=True)
    return np.repeat(gray, 3, axis=-1)
