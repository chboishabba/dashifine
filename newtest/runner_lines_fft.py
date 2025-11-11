"""Create simple FFT summaries from the dipole line lists."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Iterable, List, Sequence

import numpy as np

try:  # pragma: no cover - executed when run as a script
    from . import element_lines
    from .runner_element_lines import _parse_Z_arguments
except ImportError:  # pragma: no cover
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    import element_lines  # type: ignore
    from runner_element_lines import _parse_Z_arguments  # type: ignore


FFT_COLUMNS = [
    "Z",
    "component_index",
    "frequency_bin",
    "real",
    "imag",
    "magnitude",
    "n_lines",
    "total_intensity",
    "max_delta_energy_eV",
    "min_wavelength",
    "max_wavelength",
    "wavelength_unit",
]


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FFT_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _build_fft_rows(Z: int, unit: str, lines: Sequence[dict]) -> List[dict]:
    if not lines:
        return []
    intensities = np.array([float(line["oscillator_strength"]) for line in lines], dtype=float)
    components = np.fft.rfft(intensities)
    frequencies = np.fft.rfftfreq(len(intensities), d=1.0)
    total_intensity = float(np.sum(intensities))
    max_delta = max(float(line["delta_energy_eV"]) for line in lines)
    wavelengths = [float(line["wavelength"]) for line in lines]
    min_wavelength = min(wavelengths)
    max_wavelength = max(wavelengths)
    rows: List[dict] = []
    for index, component in enumerate(components):
        rows.append(
            {
                "Z": Z,
                "component_index": index,
                "frequency_bin": float(frequencies[index]),
                "real": float(component.real),
                "imag": float(component.imag),
                "magnitude": float(np.abs(component)),
                "n_lines": len(lines),
                "total_intensity": total_intensity,
                "max_delta_energy_eV": max_delta,
                "min_wavelength": min_wavelength,
                "max_wavelength": max_wavelength,
                "wavelength_unit": unit,
            }
        )
    return rows


def generate_fft(
    Z_values: Iterable[int],
    lmax: int,
    unit: str,
    outdir: Path,
    saveprefix: str,
) -> List[Path]:
    created_paths: List[Path] = []
    for Z in Z_values:
        lines = element_lines.build_dipole_allowed_lines(Z=Z, lmax=lmax, unit=unit)
        rows = _build_fft_rows(Z, unit, lines)
        destination = outdir / f"{saveprefix}_Z{Z}.csv"
        _write_csv(destination, rows)
        created_paths.append(destination)
    return created_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Z", type=int, default=None, help="single atomic number to compute")
    parser.add_argument(
        "--Zlist",
        type=str,
        default=None,
        help="comma-separated list of atomic numbers (e.g. 1,2,3)",
    )
    parser.add_argument("--lmax", type=int, default=3, help="maximum orbital quantum number")
    parser.add_argument(
        "--outdir", type=Path, default=Path("spectral_lines"), help="output directory"
    )
    parser.add_argument(
        "--unit",
        type=str,
        default="angstrom",
        help="wavelength unit (angstrom, nm, micron, ...)",
    )
    parser.add_argument(
        "--saveprefix",
        type=str,
        default="lines_fft",
        help="prefix for the generated CSV files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        Z_values = _parse_Z_arguments(args.Z, args.Zlist)
        generate_fft(Z_values, args.lmax, args.unit, args.outdir, args.saveprefix)
    except Exception as exc:  # pragma: no cover - consistent with the other runner
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

