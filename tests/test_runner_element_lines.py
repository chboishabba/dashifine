from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def test_element_line_runner_creates_csv(tmp_path):
    outdir = tmp_path / "spectral"
    cmd = [
        sys.executable,
        "-m",
        "newtest.runner_element_lines",
        "--Zlist",
        "1,2",
        "--lmax",
        "2",
        "--outdir",
        str(outdir),
        "--unit",
        "nm",
        "--saveprefix",
        "lines",
    ]
    subprocess.check_call(cmd)
    for Z in (1, 2):
        csv_path = outdir / f"lines_Z{Z}.csv"
        assert csv_path.exists()
        header = _read_csv_header(csv_path)
        assert header[0:3] == ["Z", "n_upper", "l_upper"]


def test_fft_runner_creates_csv(tmp_path):
    outdir = tmp_path / "spectral"
    cmd = [
        sys.executable,
        "-m",
        "newtest.runner_lines_fft",
        "--Z",
        "1",
        "--lmax",
        "2",
        "--outdir",
        str(outdir),
        "--saveprefix",
        "lines_fft",
    ]
    subprocess.check_call(cmd)
    csv_path = outdir / "lines_fft_Z1.csv"
    assert csv_path.exists()
    header = _read_csv_header(csv_path)
    assert header[0:4] == ["Z", "component_index", "frequency_bin", "real"]

