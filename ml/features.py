"""
ML2 - Network Feature Engineering

Creates the stored ML feature table used by FastAPI API4.

Feature window convention
--------------------------
For feature_timestamp = t, all features use only observations
at or before t.

Recent window:
    t-23h ... t

Baseline window:
    t-47h ... t-24h

Therefore activity_growth compares the recent 24-hour average
against the preceding 24-hour average.

No future observations are used.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "network_analytics.db"


FEATURE_COLUMNS = [
    "avg_activity",
    "activity_growth",
    "active_hours",
    "peak_ratio",
    "variability",
    "internet_share",
]


def validate_source_tables(connection: sqlite3.Connection) -> None:
    """Validate the warehouse source tables."""

    required_fact_columns = {
        "time_id",
        "grid_id",
        "sms_in",
        "sms_out",
        "call_in",
        "call_out",
        "internet_activity",
        "total_activity",
    }

    fact_rows = connection.execute(
        "PRAGMA table_info(fact_network_activity)"
    ).fetchall()

    if not fact_rows:
        raise RuntimeError(
            "Required table fact_network_activity is missing."
        )

    actual_fact_columns = {row[1] for row in fact_rows}

    missing = required_fact_columns - actual_fact_columns

    if missing:
        raise RuntimeError(
            "Missing fact_network_activity columns: "
            + ", ".join(sorted(missing))
        )

    time_rows = connection.execute(
        "PRAGMA table_info(dim_time)"
    ).fetchall()

    if not time_rows:
        raise RuntimeError(
            "Required table dim_time is missing."
        )

    actual_time_columns = {row[1] for row in time_rows}

    if "time_id" not in actual_time_columns:
        raise RuntimeError(
            "dim_time is missing required column: time_id"
        )

    if "timestamp" not in actual_time_columns:
        raise RuntimeError(
            "dim_time is missing required column: timestamp"
        )


def create_feature_table(connection: sqlite3.Connection) -> None:
    """Create the persistent ML feature table."""

    connection.execute("DROP TABLE IF EXISTS network_features")

    connection.execute(
        """
        CREATE TABLE network_features (
            grid_id INTEGER NOT NULL,
            feature_timestamp TEXT NOT NULL,

            avg_activity REAL NOT NULL,
            activity_growth REAL NOT NULL,
            active_hours INTEGER NOT NULL,
            peak_ratio REAL NOT NULL,
            variability REAL NOT NULL,
            internet_share REAL NOT NULL,

            PRIMARY KEY (grid_id, feature_timestamp)
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_network_features_grid_timestamp
        ON network_features(grid_id, feature_timestamp)
        """
    )

    connection.commit()


def build_features(connection: sqlite3.Connection) -> None:
    """
    Build ML2 features.

    For every grid and feature timestamp t:

        Recent window  = t-23h ... t
        Baseline       = t-47h ... t-24h

    A feature row is produced only when the grid has a complete
    48-hour history ending at t.
    """

    sql = """
    INSERT INTO network_features (
        grid_id,
        feature_timestamp,
        avg_activity,
        activity_growth,
        active_hours,
        peak_ratio,
        variability,
        internet_share
    )

    WITH grid_hours AS (
        SELECT
            f.grid_id,
            t.timestamp AS feature_timestamp,
            f.total_activity,
            f.internet_activity,

            ROW_NUMBER() OVER (
                PARTITION BY f.grid_id
                ORDER BY t.timestamp
            ) AS rn

        FROM fact_network_activity f

        INNER JOIN dim_time t
            ON f.time_id = t.time_id
    ),

    recent AS (
        SELECT
            current.grid_id,
            current.feature_timestamp,

            AVG(recent.total_activity) AS avg_activity,

            SUM(
                CASE
                    WHEN recent.total_activity > 0
                    THEN 1
                    ELSE 0
                END
            ) AS active_hours,

            MAX(recent.total_activity) AS peak_activity,

            AVG(
                recent.total_activity * recent.total_activity
            ) AS square_avg,

            SUM(recent.internet_activity) AS internet_total,

            SUM(recent.total_activity) AS activity_total,

            COUNT(*) AS recent_hours

        FROM grid_hours current

        INNER JOIN grid_hours recent
            ON recent.grid_id = current.grid_id
            AND recent.rn BETWEEN current.rn - 23 AND current.rn

        GROUP BY
            current.grid_id,
            current.feature_timestamp,
            current.rn
    ),

    baseline AS (
        SELECT
            current.grid_id,
            current.feature_timestamp,

            AVG(previous.total_activity) AS baseline_avg,

            COUNT(*) AS baseline_hours

        FROM grid_hours current

        INNER JOIN grid_hours previous
            ON previous.grid_id = current.grid_id
            AND previous.rn BETWEEN current.rn - 47 AND current.rn - 24

        GROUP BY
            current.grid_id,
            current.feature_timestamp,
            current.rn
    )

    SELECT
        r.grid_id,
        r.feature_timestamp,

        r.avg_activity,

        CASE
            WHEN b.baseline_avg IS NULL THEN 0.0
            WHEN b.baseline_avg = 0 THEN 0.0
            ELSE
                (r.avg_activity - b.baseline_avg)
                / b.baseline_avg
        END AS activity_growth,

        r.active_hours,

        CASE
            WHEN r.avg_activity IS NULL
                 OR r.avg_activity = 0
            THEN 0.0
            ELSE r.peak_activity / r.avg_activity
        END AS peak_ratio,

        SQRT(
            MAX(
                0.0,
                r.square_avg - (r.avg_activity * r.avg_activity)
            )
        ) AS variability,

        CASE
            WHEN r.activity_total IS NULL
                 OR r.activity_total = 0
            THEN 0.0
            ELSE r.internet_total / r.activity_total
        END AS internet_share

    FROM recent r

    INNER JOIN baseline b
        ON b.grid_id = r.grid_id
        AND b.feature_timestamp = r.feature_timestamp

    WHERE r.recent_hours = 24
      AND b.baseline_hours = 24
    """

    connection.execute(sql)
    connection.commit()


def validate_features(connection: sqlite3.Connection) -> None:
    """Validate generated feature data."""

    rows = connection.execute(
        "PRAGMA table_info(network_features)"
    ).fetchall()

    actual_columns = {row[1] for row in rows}

    required_columns = {
        "grid_id",
        "feature_timestamp",
        *FEATURE_COLUMNS,
    }

    missing = required_columns - actual_columns

    if missing:
        raise RuntimeError(
            "Feature table is missing columns: "
            + ", ".join(sorted(missing))
        )

    null_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM network_features
        WHERE
            grid_id IS NULL
            OR feature_timestamp IS NULL
            OR avg_activity IS NULL
            OR activity_growth IS NULL
            OR active_hours IS NULL
            OR peak_ratio IS NULL
            OR variability IS NULL
            OR internet_share IS NULL
        """
    ).fetchone()[0]

    if null_count:
        raise RuntimeError(
            f"Feature table contains {null_count} NULL rows."
        )

    invalid_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM network_features
        WHERE
            avg_activity < 0
            OR active_hours < 0
            OR active_hours > 24
            OR peak_ratio < 0
            OR variability < 0
            OR internet_share < 0
            OR internet_share > 1
        """
    ).fetchone()[0]

    if invalid_count:
        raise RuntimeError(
            f"Feature table contains {invalid_count} invalid rows."
        )

    duplicate_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                grid_id,
                feature_timestamp
            FROM network_features
            GROUP BY
                grid_id,
                feature_timestamp
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count:
        raise RuntimeError(
            f"Feature table contains {duplicate_count} duplicate keys."
        )

    bad_numeric_count = 0

    cursor = connection.execute(
        """
        SELECT
            avg_activity,
            activity_growth,
            peak_ratio,
            variability,
            internet_share
        FROM network_features
        """
    )

    for row in cursor:
        for value in row:
            if not math.isfinite(float(value)):
                bad_numeric_count += 1

    if bad_numeric_count:
        raise RuntimeError(
            f"Feature table contains {bad_numeric_count} "
            "NaN or infinite numeric values."
        )


def hand_check_grid(connection: sqlite3.Connection, grid_id: int = 4821) -> None:
    """
    Independently calculate avg_activity and peak_ratio for one grid.

    This is intentionally calculated using a separate query so that
    the generated feature values can be manually verified.
    """

    feature = connection.execute(
        """
        SELECT
            feature_timestamp,
            avg_activity,
            peak_ratio
        FROM network_features
        WHERE grid_id = ?
        ORDER BY feature_timestamp DESC
        LIMIT 1
        """,
        (grid_id,),
    ).fetchone()

    if feature is None:
        print(f"\nHand-check: grid {grid_id} has no feature row.")
        return

    timestamp, stored_avg, stored_peak_ratio = feature

    values = connection.execute(
        """
        SELECT
            f.total_activity
        FROM fact_network_activity f
        INNER JOIN dim_time t
            ON f.time_id = t.time_id
        WHERE
            f.grid_id = ?
            AND t.timestamp <= ?
        ORDER BY t.timestamp DESC
        LIMIT 24
        """,
        (grid_id, timestamp),
    ).fetchall()

    activities = [float(row[0]) for row in values]

    if len(activities) != 24:
        raise RuntimeError(
            f"Hand-check expected 24 observations for grid {grid_id}, "
            f"found {len(activities)}."
        )

    manual_avg = sum(activities) / 24
    manual_peak = max(activities)

    if manual_avg == 0:
        manual_peak_ratio = 0.0
    else:
        manual_peak_ratio = manual_peak / manual_avg

    print("\nHand-check:")
    print(f"  Grid: {grid_id}")
    print(f"  Timestamp: {timestamp}")
    print(f"  Stored avg_activity: {stored_avg}")
    print(f"  Manual avg_activity: {manual_avg}")
    print(f"  Stored peak_ratio: {stored_peak_ratio}")
    print(f"  Manual peak_ratio: {manual_peak_ratio}")

    if not math.isclose(
        float(stored_avg),
        manual_avg,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "avg_activity hand-check FAILED."
        )

    if not math.isclose(
        float(stored_peak_ratio),
        manual_peak_ratio,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "peak_ratio hand-check FAILED."
        )

    print("  Hand-check: PASSED")


def print_summary(connection: sqlite3.Connection) -> None:
    """Print ML2 summary."""

    row_count = connection.execute(
        "SELECT COUNT(*) FROM network_features"
    ).fetchone()[0]

    grid_count = connection.execute(
        "SELECT COUNT(DISTINCT grid_id) FROM network_features"
    ).fetchone()[0]

    timestamp_count = connection.execute(
        "SELECT COUNT(DISTINCT feature_timestamp) FROM network_features"
    ).fetchone()[0]

    min_timestamp = connection.execute(
        "SELECT MIN(feature_timestamp) FROM network_features"
    ).fetchone()[0]

    max_timestamp = connection.execute(
        "SELECT MAX(feature_timestamp) FROM network_features"
    ).fetchone()[0]

    print("=" * 70)
    print("ML2 - NETWORK FEATURE ENGINEERING")
    print("=" * 70)

    print(f"Database: {DB_PATH}")
    print(f"Feature rows: {row_count:,}")
    print(f"Grids represented: {grid_count:,}")
    print(f"Feature timestamps: {timestamp_count:,}")
    print(f"First feature timestamp: {min_timestamp}")
    print(f"Last feature timestamp: {max_timestamp}")

    print("\nFeature columns:")
    for column in FEATURE_COLUMNS:
        print(f"  {column}")

    print("  feature_timestamp")

    print("\nWindow convention:")
    print("  Recent window: t-23h ... t")
    print("  Baseline: t-47h ... t-24h")
    print("  feature_timestamp = t")
    print("  Future observations are never used")

    print("\nFeature validation: PASSED")
    print("=" * 70)


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Warehouse database not found: {DB_PATH}"
        )

    connection = sqlite3.connect(DB_PATH)

    try:
        validate_source_tables(connection)
        create_feature_table(connection)
        build_features(connection)
        validate_features(connection)
        hand_check_grid(connection, 4821)
        print_summary(connection)

    finally:
        connection.close()


if __name__ == "__main__":
    main()