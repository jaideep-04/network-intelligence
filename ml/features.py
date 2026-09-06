"""
ML2 - Network Feature Engineering

Creates the stored ML feature table used by FastAPI API4.

Feature window convention
--------------------------
For feature_timestamp = t, all features use only data available
at or before t.

Features:
    avg_activity
    activity_growth
    active_hours
    peak_ratio
    variability
    internet_share

Window:
    Recent window  = t-23h ... t       (24 hourly observations)
    Baseline window = t-47h ... t-24h  (24 hourly observations)

Therefore activity_growth compares the most recent 24 hours
against the preceding 24 hours.

No future timestamps are used.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "network_analytics.db"


def validate_source_tables(connection: sqlite3.Connection) -> None:
    """Ensure the warehouse contains the required source table/columns."""

    required_columns = {
        "time_id",
        "grid_id",
        "sms_in",
        "sms_out",
        "call_in",
        "call_out",
        "internet_activity",
        "total_activity",
    }

    rows = connection.execute(
        "PRAGMA table_info(fact_network_activity)"
    ).fetchall()

    if not rows:
        raise RuntimeError(
            "Required table fact_network_activity is missing "
            "from the analytics warehouse."
        )

    actual_columns = {row[1] for row in rows}
    missing = required_columns - actual_columns

    if missing:
        raise RuntimeError(
            "Missing required warehouse columns: "
            + ", ".join(sorted(missing))
        )


def create_feature_table(connection: sqlite3.Connection) -> None:
    """
    Create the stored network_features table.

    Only complete 48-hour histories are published because
    activity_growth requires both a recent 24-hour window and
    a preceding 24-hour baseline window.
    """

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
    Build features using SQLite window functions.

    The source data is already one row per grid/hour in
    fact_network_activity.

    We calculate:

    avg_activity:
        mean total_activity over the trailing 24 hours.

    activity_growth:
        (recent_24h_avg - previous_24h_avg)
        / previous_24h_avg

    active_hours:
        number of trailing 24-hour observations with
        total_activity > 0.

    peak_ratio:
        maximum total_activity in recent 24 hours
        divided by recent 24-hour average.

    variability:
        population standard deviation of total_activity
        over the recent 24-hour window.

    internet_share:
        sum(internet_activity) / sum(total_activity)
        over the recent 24-hour window.
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
    WITH ordered AS (
        SELECT
            f.grid_id,
            t.timestamp AS feature_timestamp,

            f.total_activity,
            f.internet_activity,

            ROW_NUMBER() OVER (
                PARTITION BY f.grid_id
                ORDER BY t.timestamp
            ) AS rn,

            AVG(f.total_activity) OVER (
                PARTITION BY f.grid_id
                ORDER BY t.timestamp
                ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
            ) AS recent_avg,

            AVG(f.total_activity) OVER (
                PARTITION BY f.grid_id
                ORDER BY t.timestamp
                ROWS BETWEEN 47 PRECEDING AND 24 PRECEDING
            ) AS previous_avg,

            COUNT(f.total_activity) OVER (
                PARTITION BY f.grid_id
                ORDER BY t.timestamp
                ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
            ) AS recent_count,

            SUM(
                CASE
                    WHEN f.total_activity > 0 THEN 1
                    ELSE 0
                END
            ) OVER (
                PARTITION BY f.grid_id
                ORDER BY t.timestamp
                ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
            ) AS recent_active_hours,

            MAX(f.total_activity) OVER (
                PARTITION BY f.grid_id
                ORDER BY t.timestamp
                ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
            ) AS recent_peak,

            SUM(f.internet_activity) OVER (
                PARTITION BY f.grid_id
                ORDER BY t.timestamp
                ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
            ) AS recent_internet,

            SUM(f.total_activity) OVER (
                PARTITION BY f.grid_id
                ORDER BY t.timestamp
                ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
            ) AS recent_total,

            AVG(
                f.total_activity * f.total_activity
            ) OVER (
                PARTITION BY f.grid_id
                ORDER BY t.timestamp
                ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
            ) AS recent_square_avg

        FROM fact_network_activity f
        JOIN dim_time t
            ON f.time_id = t.time_id
    ),

    calculated AS (
        SELECT
            grid_id,
            feature_timestamp,

            recent_avg,

            CASE
                WHEN previous_avg IS NULL THEN NULL
                WHEN previous_avg = 0 THEN 0.0
                ELSE
                    (recent_avg - previous_avg)
                    / previous_avg
            END AS growth,

            recent_active_hours,

            CASE
                WHEN recent_avg IS NULL OR recent_avg = 0 THEN 0.0
                ELSE recent_peak / recent_avg
            END AS peak_ratio_value,

            CASE
                WHEN recent_square_avg IS NULL
                     OR recent_avg IS NULL
                THEN 0.0
                ELSE
                    SQRT(
                        MAX(
                            0.0,
                            recent_square_avg
                            - (recent_avg * recent_avg)
                        )
                    )
            END AS variability_value,

            CASE
                WHEN recent_total IS NULL
                     OR recent_total = 0
                THEN 0.0
                ELSE recent_internet / recent_total
            END AS internet_share_value,

            recent_count,
            rn

        FROM ordered
    )

    SELECT
        grid_id,
        feature_timestamp,

        recent_avg AS avg_activity,

        growth AS activity_growth,

        recent_active_hours AS active_hours,

        peak_ratio_value AS peak_ratio,

        variability_value AS variability,

        internet_share_value AS internet_share

    FROM calculated

    WHERE rn >= 48
      AND recent_count = 24
      AND growth IS NOT NULL
    """

    connection.execute(sql)
    connection.commit()


def validate_features(connection: sqlite3.Connection) -> None:
    """Run data-quality checks on the generated feature table."""

    required_columns = {
        "grid_id",
        "feature_timestamp",
        "avg_activity",
        "activity_growth",
        "active_hours",
        "peak_ratio",
        "variability",
        "internet_share",
    }

    rows = connection.execute(
        "PRAGMA table_info(network_features)"
    ).fetchall()

    actual_columns = {row[1] for row in rows}

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
            avg_activity IS NULL
            OR activity_growth IS NULL
            OR active_hours IS NULL
            OR peak_ratio IS NULL
            OR variability IS NULL
            OR internet_share IS NULL
            OR feature_timestamp IS NULL
        """
    ).fetchone()[0]

    if null_count:
        raise RuntimeError(
            f"Feature table contains {null_count} rows with NULL features."
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
            f"Feature table contains {invalid_count} invalid feature rows."
        )

    duplicate_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT grid_id, feature_timestamp
            FROM network_features
            GROUP BY grid_id, feature_timestamp
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count:
        raise RuntimeError(
            f"Feature table contains {duplicate_count} duplicate keys."
        )


def print_summary(connection: sqlite3.Connection) -> None:
    """Print useful ML2 output information."""

    row_count = connection.execute(
        "SELECT COUNT(*) FROM network_features"
    ).fetchone()[0]

    grid_count = connection.execute(
        "SELECT COUNT(DISTINCT grid_id) FROM network_features"
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
    print(f"First feature timestamp: {min_timestamp}")
    print(f"Last feature timestamp: {max_timestamp}")

    print("\nFeature columns:")
    print("  avg_activity")
    print("  activity_growth")
    print("  active_hours")
    print("  peak_ratio")
    print("  variability")
    print("  internet_share")
    print("  feature_timestamp")

    print("\nWindow convention:")
    print("  Recent window: previous 24 hours ending at t")
    print("  Baseline: preceding 24 hours")
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
        print_summary(connection)

    finally:
        connection.close()


if __name__ == "__main__":
    main()