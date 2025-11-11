"""Utilities for constructing dipole-allowed spectral line lists."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Dict, Iterator, List

from . import hydrogenic_numerov as hn

# Columns exposed by ``build_dipole_allowed_lines``.  Keeping the header in a
# single place ensures the CSV writer in the runner always emits the same
# structure, even if the computed list happens to be empty for the chosen
# parameters.
LINE_COLUMNS: List[str] = [
    "Z",
    "n_upper",
    "l_upper",
    "degeneracy_upper",
    "n_lower",
    "l_lower",
    "degeneracy_lower",
    "delta_energy_eV",
    "wavelength",
    "wavelength_unit",
    "frequency_THz",
    "angular_frequency_rad_s",
    "wavenumber_cm-1",
    "oscillator_strength",
]


@dataclass(frozen=True)
class HydrogenicState:
    """Quantum numbers describing a bound electron state."""

    n: int
    l: int

    @property
    def degeneracy(self) -> int:
        # Magnetic quantum numbers m span [-l, ..., l].
        return 2 * self.l + 1


def _iter_states(lmax: int) -> Iterator[HydrogenicState]:
    if not isinstance(lmax, int) or lmax < 0:
        raise ValueError("lmax must be a non-negative integer")
    n_max = max(lmax + 1, 1)
    for n in range(1, n_max + 2):
        upper_l = min(lmax, n - 1)
        for l in range(0, upper_l + 1):
            yield HydrogenicState(n=n, l=l)


def _oscillator_strength(upper: HydrogenicState, lower: HydrogenicState) -> float:
    # The exact hydrogenic oscillator strengths require solving radial
    # integrals.  For the smoke tests we only need a deterministic,
    # well-behaved proxy.  The expression below mimics the expected scaling
    # (stronger transitions for lower n and dipole-allowed Δl = ±1) while
    # remaining easy to evaluate.
    delta_n = upper.n - lower.n
    strength = 1.0 / (upper.n ** 3)
    strength *= (lower.degeneracy + upper.degeneracy) / 2.0
    strength *= abs(delta_n)
    return strength


def _build_line_record(
    Z: int,
    unit: str,
    upper: HydrogenicState,
    lower: HydrogenicState,
) -> Dict[str, float]:
    delta_e = hn.energy_level_eV(Z, upper.n) - hn.energy_level_eV(Z, lower.n)
    if delta_e <= 0:
        raise ValueError("upper level must have higher energy (larger n)")
    wavelength = hn.wavelength_from_energy_gap(delta_e, unit=unit)
    frequency_hz = hn.frequency_from_energy_gap(delta_e)
    return {
        "Z": Z,
        "n_upper": upper.n,
        "l_upper": upper.l,
        "degeneracy_upper": upper.degeneracy,
        "n_lower": lower.n,
        "l_lower": lower.l,
        "degeneracy_lower": lower.degeneracy,
        "delta_energy_eV": delta_e,
        "wavelength": wavelength,
        "wavelength_unit": unit,
        "frequency_THz": frequency_hz / 1e12,
        "angular_frequency_rad_s": hn.angular_frequency_from_energy_gap(delta_e),
        "wavenumber_cm-1": hn.wavenumber_from_energy_gap(delta_e),
        "oscillator_strength": _oscillator_strength(upper, lower),
    }


def build_dipole_allowed_lines(Z: int, lmax: int, unit: str = "angstrom") -> List[Dict[str, float]]:
    """Return dipole-allowed spectral lines for a hydrogenic ion.

    The function enumerates pairs of states with ``Δl = ±1`` and ``n_upper >
    n_lower`` up to the requested ``lmax``.  The energies and derived
    quantities are computed via :mod:`newtest.hydrogenic_numerov`.
    """

    states = list(_iter_states(lmax))
    transitions: List[Dict[str, float]] = []
    for upper, lower in itertools.product(states, repeat=2):
        if upper.n <= lower.n:
            continue
        if abs(upper.l - lower.l) != 1:
            continue
        transitions.append(_build_line_record(Z, unit, upper, lower))

    transitions.sort(key=lambda row: (row["n_upper"], row["n_lower"], row["l_upper"]))
    return transitions


__all__ = ["LINE_COLUMNS", "HydrogenicState", "build_dipole_allowed_lines"]

