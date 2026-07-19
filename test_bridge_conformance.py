"""Golden conformance check for the Dashifine ↔ TextGraphs bridge.

This test intentionally separates two claims:

1. transported Dashifine aggregates must match their Julia/TextGraphs copies;
2. graph-derived metrics and correlations are regression-checked, but are not
   promoted as independently discovered semantic invariants.

Run with:
    python test_bridge_conformance.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "bridge_conformance_golden.json"
BASELINE = ROOT / "textgraphs_graph_props.csv"
AUGMENTED = ROOT / "textgraphs_graph_props_augmented.csv"
CORRELATIONS = ROOT / "bridge_correlation.csv"
VARIANTS = ROOT / "textgraphs_variant_props.csv"


class ConformanceError(AssertionError):
    """Raised when a bridge artifact drifts from the canonical fixture."""


def read_key_value_csv(path: Path) -> Dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {row["key"]: float(row["value"]) for row in rows}


def read_correlation_csv(path: Path) -> Dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {row["pair"]: float(row["pearson"]) for row in rows}


def assert_close_map(
    label: str,
    actual: Dict[str, float],
    expected: Dict[str, float],
    tolerance: float,
) -> None:
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ConformanceError(
            f"{label} keys drifted: missing={missing}, extra={extra}"
        )

    for key, expected_value in expected.items():
        actual_value = actual[key]
        if not math.isclose(
            actual_value,
            expected_value,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise ConformanceError(
                f"{label}.{key} drifted: actual={actual_value!r}, "
                f"expected={expected_value!r}"
            )


def read_variant_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one_variant_value(
    rows: Iterable[dict[str, str]], variant: str, metric: str, column: str
) -> float:
    matches = [
        row
        for row in rows
        if row["variant"] == variant and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise ConformanceError(
            f"expected one row for variant={variant!r}, metric={metric!r}; "
            f"found {len(matches)}"
        )
    return float(matches[0][column])


def main() -> None:
    required = [GOLDEN, BASELINE, AUGMENTED, CORRELATIONS, VARIANTS]
    missing = [str(path.name) for path in required if not path.exists()]
    if missing:
        raise ConformanceError(f"missing bridge artifacts: {missing}")

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    tolerance = float(golden["tolerance"])

    assert_close_map(
        "baseline_props",
        read_key_value_csv(BASELINE),
        golden["baseline_props"],
        tolerance,
    )
    assert_close_map(
        "augmented_props",
        read_key_value_csv(AUGMENTED),
        golden["augmented_props"],
        tolerance,
    )
    assert_close_map(
        "correlations",
        read_correlation_csv(CORRELATIONS),
        golden["correlations"],
        tolerance,
    )

    variant_rows = read_variant_rows(VARIANTS)
    contract = golden["variant_contract"]
    actual_variants = sorted({row["variant"] for row in variant_rows})
    expected_variants = sorted(contract["expected_variants"])
    if actual_variants != expected_variants:
        raise ConformanceError(
            f"variant set drifted: actual={actual_variants}, "
            f"expected={expected_variants}"
        )

    expected_count = int(contract["expected_metric_count_per_variant"])
    for variant in expected_variants:
        count = sum(row["variant"] == variant for row in variant_rows)
        if count != expected_count:
            raise ConformanceError(
                f"variant {variant!r} has {count} metrics; expected {expected_count}"
            )

    checks = [
        (
            one_variant_value(variant_rows, "baseline", "edge_count", "value"),
            float(contract["baseline_edge_count"]),
            "baseline edge_count",
        ),
        (
            one_variant_value(
                variant_rows, "ternary_l1_le_1", "edge_count", "added_edges"
            ),
            float(contract["ternary_l1_le_1_added_edges"]),
            "ternary_l1_le_1 added_edges",
        ),
        (
            one_variant_value(
                variant_rows, "ternary_l1_le_2", "edge_count", "added_edges"
            ),
            float(contract["ternary_l1_le_2_added_edges"]),
            "ternary_l1_le_2 added_edges",
        ),
    ]
    for actual, expected, label in checks:
        if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
            raise ConformanceError(
                f"{label} drifted: actual={actual!r}, expected={expected!r}"
            )

    for pair in golden["interpretation_boundary"]["exact_transport_fields"]:
        value = golden["correlations"][pair]
        if not math.isclose(value, 1.0, rel_tol=tolerance, abs_tol=tolerance):
            raise ConformanceError(
                f"transport parity failed for {pair!r}: correlation={value!r}"
            )

    print(
        "bridge conformance passed: baseline, augmented, correlations, "
        "variant contract, and exact R/z transport parity"
    )


if __name__ == "__main__":
    main()
