"""Command line interface for generating spectral line CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import List, Sequence

try:  # pragma: no cover - executed when run as a script
    from . import element_lines
except ImportError:  # pragma: no cover
    # When executed as ``python newtest/runner_element_lines.py`` the package
    # relative import is not available.  We gracefully fall back to a direct
    # lookup.
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    import element_lines  # type: ignore


def _parse_Z_arguments(Z: int | None, Zlist: str | None) -> List[int]:
    values: List[int] = []
    if Z is not None:
        if Z <= 0:
            raise ValueError("--Z must be a positive integer")
        values.append(Z)
    if Zlist:
        for chunk in Zlist.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            value = int(chunk)
            if value <= 0:
                raise ValueError("--Zlist entries must be positive integers")
            values.append(value)
    if not values:
        raise ValueError("at least one of --Z or --Zlist must be provided")
    # Remove duplicates while preserving order.
    seen = set()
    unique_values: List[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        fieldnames = element_lines.LINE_COLUMNS
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def generate_lines(
    Z_values: Sequence[int],
    lmax: int,
    unit: str,
    outdir: Path,
    saveprefix: str,
) -> List[Path]:
    created_paths: List[Path] = []
    for Z in Z_values:
        rows = element_lines.build_dipole_allowed_lines(Z=Z, lmax=lmax, unit=unit)
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
        default="lines",
        help="prefix for the generated CSV files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        Z_values = _parse_Z_arguments(args.Z, args.Zlist)
        generate_lines(Z_values, args.lmax, args.unit, args.outdir, args.saveprefix)
    except Exception as exc:  # pragma: no cover - argparse already tested indirectly
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    sys.exit(main())

