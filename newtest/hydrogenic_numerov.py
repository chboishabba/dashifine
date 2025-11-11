"""Hydrogenic helper routines used by the spectral tooling.

The real project historically relied on Numerov integration of the
hydrogenic radial equation.  For the purposes of the automated tests we
only need lightweight, well-behaved helpers that expose the same public
surface: the energy levels of the hydrogenic atom and conversions between
energy and wavelength/frequency units.  The expressions implemented here
follow the textbook formulas for an electron bound in a Coulomb
potential.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

# Fundamental constants.  They are gathered in a small dataclass for
# clarity and to make the unit conversions explicit.


@dataclass(frozen=True)
class _Constants:
    planck_constant: float = 6.626_070_15e-34  # J*s
    reduced_planck_constant: float = 1.054_571_817e-34  # J*s
    speed_of_light: float = 299_792_458.0  # m/s
    electron_volt: float = 1.602_176_634e-19  # J
    rydberg_energy: float = 13.605_693_009  # eV


CONSTANTS = _Constants()


# Unit conversion factors from meters to the requested unit.
_WAVELENGTH_UNITS: Dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "cm": 1e-2,
    "centimeter": 1e-2,
    "centimeters": 1e-2,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "millimeters": 1e-3,
    "micron": 1e-6,
    "microns": 1e-6,
    "um": 1e-6,
    "µm": 1e-6,
    "nm": 1e-9,
    "nanometer": 1e-9,
    "nanometers": 1e-9,
    "angstrom": 1e-10,
    "angstroms": 1e-10,
}


def _normalise_unit(unit: str) -> str:
    try:
        return unit.strip().lower()
    except AttributeError as exc:  # pragma: no cover - defensive programming
        raise TypeError("unit must be a string") from exc


def energy_level_eV(Z: int, n: int) -> float:
    """Return the hydrogenic energy level for ``(Z, n)`` in electron volts.

    The familiar closed-form expression is

    .. math:: E_n = -R Z^2 / n^2

    where :math:`R` is the Rydberg energy.  The function guards against
    non-physical inputs so that callers do not have to duplicate the same
    validation logic.
    """

    if not isinstance(Z, int) or Z <= 0:
        raise ValueError("Z must be a positive integer")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    return -CONSTANTS.rydberg_energy * (Z ** 2) / (n ** 2)


def energy_level_J(Z: int, n: int) -> float:
    """Return the hydrogenic energy level in Joules."""

    return energy_level_eV(Z, n) * CONSTANTS.electron_volt


def energy_gap_eV(upper_n: int, lower_n: int, Z: int) -> float:
    """Return the (positive) energy gap between two levels in eV."""

    upper = energy_level_eV(Z, upper_n)
    lower = energy_level_eV(Z, lower_n)
    gap = upper - lower
    if gap <= 0:
        raise ValueError("upper level must be less bound than lower level")
    return gap


def wavelength_from_energy_gap(delta_eV: float, unit: str = "angstrom") -> float:
    """Convert an energy gap (in eV) to a wavelength in the desired unit."""

    if delta_eV <= 0:
        raise ValueError("energy gap must be positive")
    unit_key = _normalise_unit(unit)
    try:
        scale = _WAVELENGTH_UNITS[unit_key]
    except KeyError as exc:
        raise ValueError(f"unsupported wavelength unit: {unit}") from exc

    delta_joule = delta_eV * CONSTANTS.electron_volt
    wavelength_m = CONSTANTS.planck_constant * CONSTANTS.speed_of_light / delta_joule
    return wavelength_m / scale


def frequency_from_energy_gap(delta_eV: float) -> float:
    """Convert an energy gap (in eV) to a frequency in Hz."""

    if delta_eV <= 0:
        raise ValueError("energy gap must be positive")
    delta_joule = delta_eV * CONSTANTS.electron_volt
    return delta_joule / CONSTANTS.planck_constant


def angular_frequency_from_energy_gap(delta_eV: float) -> float:
    """Convert an energy gap to an angular frequency in rad/s."""

    return 2.0 * math.pi * frequency_from_energy_gap(delta_eV)


def wavenumber_from_energy_gap(delta_eV: float) -> float:
    """Return the wavenumber (cm⁻¹) associated with an energy gap."""

    delta_joule = delta_eV * CONSTANTS.electron_volt
    wavelength_m = CONSTANTS.planck_constant * CONSTANTS.speed_of_light / delta_joule
    wavelength_cm = wavelength_m * 100.0
    return 1.0 / wavelength_cm


__all__ = [
    "CONSTANTS",
    "energy_level_eV",
    "energy_level_J",
    "energy_gap_eV",
    "wavelength_from_energy_gap",
    "frequency_from_energy_gap",
    "angular_frequency_from_energy_gap",
    "wavenumber_from_energy_gap",
]

