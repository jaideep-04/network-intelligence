"""
ML6 - Top-20 Operational Attention Report

Reads the latest network_risk_scores and network_features
from the analytics warehouse and produces:
    reports/ml6_top20_operational_attention.csv
    reports/ml6_top20_operational_attention.md

The report is an operational investigation list.
It does not diagnose confirmed network faults.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = (
    PROJECT_ROOT
    / "warehouse"
    / "network_analytics.db"
)

REPORT_DIR = PROJECT_ROOT / "reports"

CSV_PATH = (
    REPORT_DIR
    / "ml6_top20_operational_attention.csv"
)

MD_PATH = (
    REPORT_DIR
    / "ml6_top20_operational_attention.md"
)


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def main() -> None:

    print("=" * 75)
    print("ML6 - TOP-20 OPERATIONAL ATTENTION REPORT")
    print("=" * 75)

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT
                r.grid_id,
                r.feature_timestamp,
                r.risk_score,
                r.risk_level,
                r.model_version,
                f.avg_activity,
                f.activity_growth,
                f.active_hours,
                f.peak_ratio,
                f.variability,
                f.internet_share
            FROM network_risk_scores r
            JOIN network_features f
              ON r.grid_id = f.grid_id
             AND r.feature_timestamp = f.feature_timestamp
            ORDER BY
                r.risk_score DESC,
                r.grid_id ASC
            LIMIT 20
            """
        ).fetchall()

    finally:
        connection.close()

    if len(rows) != 20:
        raise RuntimeError(
            f"Expected 20 rows, found {len(rows)}"
        )

    report_rows = []

    for rank, row in enumerate(rows, start=1):

        risk_level = str(
            row["risk_level"]
        )

        if risk_level == "HIGH":
            reason = (
                "High model-estimated unusual "
                "activity risk."
            )

            attention = (
                "Investigate the grid activity pattern "
                "and review recent observations. "
                "This is an investigation signal, "
                "not a confirmed network fault."
            )

        elif risk_level == "MEDIUM":
            reason = (
                "Medium model-estimated unusual "
                "activity risk."
            )

            attention = (
                "Review the grid activity pattern and "
                "recent observations when operationally "
                "appropriate."
            )

        else:
            reason = (
                "Low model-estimated unusual "
                "activity risk."
            )

            attention = (
                "No immediate model-driven attention "
                "is indicated; continue normal monitoring."
            )

        report_rows.append(
            {
                "rank": rank,
                "grid_id": int(row["grid_id"]),
                "feature_timestamp": row[
                    "feature_timestamp"
                ],
                "risk_score": float(
                    row["risk_score"]
                ),
                "risk_level": risk_level,
                "model_version": row[
                    "model_version"
                ],
                "reason": reason,
                "attention": attention,
            }
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_columns = [
        "rank",
        "grid_id",
        "feature_timestamp",
        "risk_score",
        "risk_level",
        "model_version",
        "reason",
        "attention",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=csv_columns,
        )

        writer.writeheader()

        for report_row in report_rows:
            writer.writerow(report_row)

    # --------------------------------------------------------
    # Markdown
    # --------------------------------------------------------

    lines = [
        "# ML6 Top-20 Operational Attention Report",
        "",
        "This report ranks the latest grid scores produced "
        "by the ML6 batch-scoring stage.",
        "",
        f"**Model version:** "
        f"{report_rows[0]['model_version']}",
        "",
        f"**Feature timestamp:** "
        f"{report_rows[0]['feature_timestamp']}",
        "",
        "**Interpretation:** Higher scores indicate that "
        "the grid activity pattern should be investigated. "
        "The score is an investigation signal and does not "
        "represent a confirmed network fault.",
        "",
        "| Rank | Grid | Risk Score | Level | Reason | Attention |",
        "|---:|---:|---:|---|---|---|",
    ]

    for row in report_rows:

        lines.append(
            f"| {row['rank']} "
            f"| {row['grid_id']} "
            f"| {row['risk_score']:.6f} "
            f"| {row['risk_level']} "
            f"| {row['reason']} "
            f"| {row['attention']} |"
        )

    lines.extend(
        [
            "",
            "## Validation",
            "",
            "- Exactly 20 grids are included.",
            "- Scores are ordered descending.",
            "- Every row contains a model version.",
            "- Risk levels come from the ML6 scoring thresholds.",
            "- The report uses investigation language.",
            "- The report does not diagnose confirmed faults.",
        ]
    )

    MD_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console validation
    # --------------------------------------------------------

    print()
    print("Top 20 rows:", len(report_rows))
    print()
    print(
        f"CSV report : {CSV_PATH}"
    )
    print(
        f"Markdown   : {MD_PATH}"
    )

    print()
    print("-" * 75)
    print(
        f"{'Rank':<6}"
        f"{'Grid':<8}"
        f"{'Score':<14}"
        f"{'Level':<10}"
    )
    print("-" * 75)

    for row in report_rows:

        print(
            f"{row['rank']:<6}"
            f"{row['grid_id']:<8}"
            f"{row['risk_score']:<14.6f}"
            f"{row['risk_level']:<10}"
        )

    print("-" * 75)

    # Final validation
    scores = [
        row["risk_score"]
        for row in report_rows
    ]

    if scores != sorted(
        scores,
        reverse=True,
    ):
        raise RuntimeError(
            "Report scores are not sorted descending."
        )

    missing_versions = [
        row
        for row in report_rows
        if not row["model_version"]
    ]

    if missing_versions:
        raise RuntimeError(
            "One or more report rows have no model version."
        )

    print()
    print("REPORT VALIDATION: PASSED")


if __name__ == "__main__":
    main()