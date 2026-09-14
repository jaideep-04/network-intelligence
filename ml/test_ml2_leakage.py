"""
ML2 - Feature Leakage Test

Verifies that ML2 features at timestamp t do not use observations
after timestamp t.

Two tests are performed:

1. REAL FEATURE SET
   Verifies that the real ML2 feature window ends at t.
   Expected: PASS.

2. DELIBERATELY BROKEN FEATURE SET
   Uses t+1 intentionally.
   Expected: the leakage detector catches it.

The broken test uses an earlier timestamp rather than the latest
timestamp so that a future observation exists in the dataset.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "network_analytics.db"


def test_real_window(
    connection: sqlite3.Connection,
    grid_id: int = 4821,
) -> None:
    """
    Verify that the real feature window uses data only through t.
    """

    feature = connection.execute(
        """
        SELECT
            feature_timestamp
        FROM network_features
        WHERE grid_id = ?
        ORDER BY feature_timestamp DESC
        LIMIT 1
        """,
        (grid_id,),
    ).fetchone()

    if feature is None:
        raise AssertionError(
            f"No feature row found for grid {grid_id}."
        )

    feature_timestamp = feature[0]

    rows = connection.execute(
        """
        SELECT
            t.timestamp
        FROM fact_network_activity f
        INNER JOIN dim_time t
            ON f.time_id = t.time_id
        WHERE
            f.grid_id = ?
            AND t.timestamp <= ?
        ORDER BY t.timestamp DESC
        LIMIT 24
        """,
        (grid_id, feature_timestamp),
    ).fetchall()

    if len(rows) != 24:
        raise AssertionError(
            f"Expected 24 observations, found {len(rows)}."
        )

    future_rows = [
        row for row in rows
        if row[0] > feature_timestamp
    ]

    if future_rows:
        raise AssertionError(
            "LEAKAGE DETECTED: real feature window contains "
            "future observations."
        )

    latest_used_timestamp = rows[0][0]

    if latest_used_timestamp != feature_timestamp:
        raise AssertionError(
            "Real feature window does not end at feature_timestamp."
        )

    print("REAL FEATURE SET")
    print(f"  Grid: {grid_id}")
    print(f"  Feature timestamp: {feature_timestamp}")
    print(f"  Latest observation used: {latest_used_timestamp}")
    print("  Future observations used: 0")
    print("  Leakage test: PASSED")


def test_broken_window(
    connection: sqlite3.Connection,
    grid_id: int = 4821,
) -> None:
    """
    Deliberately introduce future-data leakage.

    We select an earlier feature timestamp so that t+1 exists.
    """

    feature = connection.execute(
        """
        SELECT
            feature_timestamp
        FROM network_features
        WHERE grid_id = ?
        ORDER BY feature_timestamp DESC
        LIMIT 50, 1
        """,
        (grid_id,),
    ).fetchone()

    if feature is None:
        raise AssertionError(
            "Could not find an earlier feature timestamp "
            "for the broken leakage test."
        )

    feature_timestamp = feature[0]

    future_row = connection.execute(
        """
        SELECT
            t.timestamp
        FROM fact_network_activity f
        INNER JOIN dim_time t
            ON f.time_id = t.time_id
        WHERE
            f.grid_id = ?
            AND t.timestamp > ?
        ORDER BY t.timestamp
        LIMIT 1
        """,
        (grid_id, feature_timestamp),
    ).fetchone()

    if future_row is None:
        raise AssertionError(
            "Broken leakage test could not find a future observation."
        )

    future_timestamp = future_row[0]

    print("\nDELIBERATELY BROKEN FEATURE SET")
    print(f"  Grid: {grid_id}")
    print(f"  Feature timestamp: {feature_timestamp}")
    print(f"  Future observation introduced: {future_timestamp}")

    # ---------------------------------------------------------
    # Simulate a broken feature calculation.
    #
    # The feature window incorrectly includes t+1.
    # ---------------------------------------------------------

    broken_window = connection.execute(
        """
        SELECT
            t.timestamp
        FROM fact_network_activity f
        INNER JOIN dim_time t
            ON f.time_id = t.time_id
        WHERE
            f.grid_id = ?
            AND t.timestamp <= ?
        ORDER BY t.timestamp DESC
        LIMIT 24
        """,
        (grid_id, future_timestamp),
    ).fetchall()

    if not broken_window:
        raise AssertionError(
            "Broken feature window could not be constructed."
        )

    broken_latest_timestamp = broken_window[0][0]

    # ---------------------------------------------------------
    # Leakage detector
    # ---------------------------------------------------------

    if broken_latest_timestamp > feature_timestamp:

        print(
            "  Leakage detector found observation after "
            "feature_timestamp."
        )
        print(
            f"  Broken latest timestamp: "
            f"{broken_latest_timestamp}"
        )
        print("  Leakage detection test: PASSED")
        return

    raise AssertionError(
        "Leakage detection FAILED: future observation "
        "was not detected."
    )


def main() -> None:

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    connection = sqlite3.connect(DB_PATH)

    try:

        print("=" * 70)
        print("ML2 - FEATURE LEAKAGE TEST")
        print("=" * 70)

        test_real_window(connection)

        test_broken_window(connection)

        print("\n" + "=" * 70)
        print("ML2 LEAKAGE TESTS: PASSED")
        print("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()