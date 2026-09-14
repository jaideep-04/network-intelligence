"""
ML6 - Batch Score All Grids

Reads the latest ML2 feature row for every grid, scores the grids using
the ML4 selected model, and publishes the results to network_risk_scores.

The prediction represents operational attention for an unusual activity
pattern in the next hourly interval. It does not represent a confirmed
network fault, congestion, or capacity problem.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "warehouse" / "network_analytics.db"
MODEL_PATH = PROJECT_ROOT / "ml" / "artifacts" / "ml4_best_model.pkl"


FEATURE_COLUMNS = [
    "avg_activity",
    "activity_growth",
    "active_hours",
    "peak_ratio",
    "variability",
    "internet_share",
]


RISK_SCORE_HIGH = 0.66
RISK_SCORE_MEDIUM = 0.33


def load_model_artifact() -> dict:
    """Load and validate the ML4 model artifact."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ML model artifact not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict):
        raise RuntimeError(
            "ML model artifact must be a dictionary."
        )

    required_keys = [
        "model",
        "model_name",
        "model_version",
        "feature_columns",
    ]

    missing = [
        key
        for key in required_keys
        if key not in artifact
    ]

    if missing:
        raise RuntimeError(
            "ML model artifact is missing required keys: "
            + ", ".join(missing)
        )

    if artifact["feature_columns"] != FEATURE_COLUMNS:
        raise RuntimeError(
            "ML model feature columns do not match ML6 feature contract.\n"
            f"Expected: {FEATURE_COLUMNS}\n"
            f"Found: {artifact['feature_columns']}"
        )

    model = artifact["model"]

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "Selected ML model does not support predict_proba()."
        )

    return artifact


def validate_feature_table(connection: sqlite3.Connection) -> None:
    """Verify that ML2 features exist before scoring."""

    table_info = connection.execute(
        "PRAGMA table_info(network_features)"
    ).fetchall()

    if not table_info:
        raise RuntimeError(
            "network_features table does not exist. "
            "Run feature generation before ML6 scoring."
        )

    actual_columns = {
        row[1]
        for row in table_info
    }

    required_columns = {
        "grid_id",
        "feature_timestamp",
        *FEATURE_COLUMNS,
    }

    missing = required_columns - actual_columns

    if missing:
        raise RuntimeError(
            "network_features is missing required columns: "
            + ", ".join(sorted(missing))
        )

    row_count = connection.execute(
        "SELECT COUNT(*) FROM network_features"
    ).fetchone()[0]

    if row_count == 0:
        raise RuntimeError(
            "network_features exists but contains no rows. "
            "Run feature generation before ML6 scoring."
        )


def read_latest_features(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Read the latest feature row for every grid.

    ML6 publishes the current risk state, so one latest feature row
    is scored for each grid.
    """

    sql = """
    SELECT
        grid_id,
        feature_timestamp,
        avg_activity,
        activity_growth,
        active_hours,
        peak_ratio,
        variability,
        internet_share
    FROM (
        SELECT
            grid_id,
            feature_timestamp,
            avg_activity,
            activity_growth,
            active_hours,
            peak_ratio,
            variability,
            internet_share,
            ROW_NUMBER() OVER (
                PARTITION BY grid_id
                ORDER BY feature_timestamp DESC
            ) AS rn
        FROM network_features
    )
    WHERE rn = 1
    ORDER BY grid_id
    """

    dataframe = pd.read_sql_query(
        sql,
        connection,
    )

    if dataframe.empty:
        raise RuntimeError(
            "No latest feature rows were available for ML6 scoring."
        )

    return dataframe


def validate_input_features(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the batch scoring input."""

    expected_columns = [
        "grid_id",
        "feature_timestamp",
        *FEATURE_COLUMNS,
    ]

    missing = [
        column
        for column in expected_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise RuntimeError(
            "ML6 input is missing columns: "
            + ", ".join(missing)
        )

    if dataframe["grid_id"].isnull().any():
        raise RuntimeError(
            "ML6 input contains NULL grid_id values."
        )

    if dataframe["feature_timestamp"].isnull().any():
        raise RuntimeError(
            "ML6 input contains NULL feature_timestamp values."
        )

    if dataframe[FEATURE_COLUMNS].isnull().any().any():
        raise RuntimeError(
            "ML6 input contains NULL feature values."
        )

    for column in FEATURE_COLUMNS:
        numeric_values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if numeric_values.isnull().any():
            raise RuntimeError(
                f"ML6 feature column '{column}' contains "
                "non-numeric values."
            )

        if not numeric_values.map(math.isfinite).all():
            raise RuntimeError(
                f"ML6 feature column '{column}' contains "
                "NaN or infinite values."
            )

    if (
        dataframe["active_hours"] < 0
    ).any() or (
        dataframe["active_hours"] > 24
    ).any():
        raise RuntimeError(
            "active_hours contains values outside 0-24."
        )

    if (dataframe["avg_activity"] < 0).any():
        raise RuntimeError(
            "avg_activity contains negative values."
        )

    if (dataframe["peak_ratio"] < 0).any():
        raise RuntimeError(
            "peak_ratio contains negative values."
        )

    if (dataframe["variability"] < 0).any():
        raise RuntimeError(
            "variability contains negative values."
        )

    if (
        dataframe["internet_share"] < 0
    ).any() or (
        dataframe["internet_share"] > 1
    ).any():
        raise RuntimeError(
            "internet_share contains values outside 0-1."
        )

    duplicate_count = dataframe.duplicated(
        subset=["grid_id", "feature_timestamp"]
    ).sum()

    if duplicate_count:
        raise RuntimeError(
            f"ML6 input contains {duplicate_count} duplicate "
            "grid/timestamp rows."
        )


def score_batch(
    dataframe: pd.DataFrame,
    artifact: dict,
) -> pd.DataFrame:
    """Score all latest grid feature rows in one batch."""

    model = artifact["model"]
    model_version = str(artifact["model_version"])

    model_input = dataframe[
        FEATURE_COLUMNS
    ].copy()

    probabilities = model.predict_proba(model_input)

    classes = list(model.classes_)

    if 1 not in classes:
        raise RuntimeError(
            "ML model does not contain positive class 1."
        )

    positive_class_index = classes.index(1)

    risk_scores = probabilities[
        :,
        positive_class_index,
    ]

    result = dataframe[
        [
            "grid_id",
            "feature_timestamp",
        ]
    ].copy()

    result["risk_score"] = risk_scores.astype(float)

    result["risk_level"] = result[
        "risk_score"
    ].apply(
        lambda score: (
            "HIGH"
            if score >= RISK_SCORE_HIGH
            else (
                "MEDIUM"
                if score >= RISK_SCORE_MEDIUM
                else "LOW"
            )
        )
    )

    result["model_version"] = model_version

    return result


def create_risk_table(
    connection: sqlite3.Connection,
) -> None:
    """Create the persistent ML6 risk-score table."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS network_risk_scores (
            grid_id INTEGER NOT NULL,
            feature_timestamp TEXT NOT NULL,
            risk_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            model_version TEXT NOT NULL,

            PRIMARY KEY (
                grid_id,
                feature_timestamp
            )
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_network_risk_scores_timestamp
        ON network_risk_scores(feature_timestamp)
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_network_risk_scores_level_score
        ON network_risk_scores(risk_level, risk_score DESC)
        """
    )

    connection.commit()


def publish_scores(
    connection: sqlite3.Connection,
    scores: pd.DataFrame,
) -> None:
    """
    Publish the batch results.

    INSERT OR REPLACE makes the stage repeatable and idempotent
    for the same grid/timestamp/model output.
    """

    rows = [
        (
            int(row.grid_id),
            str(row.feature_timestamp),
            float(row.risk_score),
            str(row.risk_level),
            str(row.model_version),
        )
        for row in scores.itertuples(index=False)
    ]

    connection.executemany(
        """
        INSERT OR REPLACE INTO network_risk_scores (
            grid_id,
            feature_timestamp,
            risk_score,
            risk_level,
            model_version
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )

    connection.commit()


def validate_published_scores(
    connection: sqlite3.Connection,
    expected_count: int,
) -> None:
    """Validate the published ML6 scores."""

    actual_count = connection.execute(
        "SELECT COUNT(*) FROM network_risk_scores"
    ).fetchone()[0]

    if actual_count < expected_count:
        raise RuntimeError(
            "ML6 publication validation failed: "
            f"expected at least {expected_count} rows, "
            f"found {actual_count}."
        )

    invalid_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM network_risk_scores
        WHERE
            grid_id IS NULL
            OR feature_timestamp IS NULL
            OR risk_score IS NULL
            OR risk_level IS NULL
            OR model_version IS NULL
            OR risk_score < 0
            OR risk_score > 1
            OR risk_level NOT IN (
                'LOW',
                'MEDIUM',
                'HIGH'
            )
        """
    ).fetchone()[0]

    if invalid_count:
        raise RuntimeError(
            f"ML6 published {invalid_count} invalid risk rows."
        )

    duplicate_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                grid_id,
                feature_timestamp
            FROM network_risk_scores
            GROUP BY
                grid_id,
                feature_timestamp
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count:
        raise RuntimeError(
            f"ML6 found {duplicate_count} duplicate "
            "risk-score keys."
        )

    missing_model_version = connection.execute(
        """
        SELECT COUNT(*)
        FROM network_risk_scores
        WHERE TRIM(model_version) = ''
        """
    ).fetchone()[0]

    if missing_model_version:
        raise RuntimeError(
            "ML6 found risk scores without model_version."
        )


def print_summary(
    connection: sqlite3.Connection,
    scores: pd.DataFrame,
    artifact: dict,
) -> None:
    """Print an ML6 execution summary."""

    latest_timestamp = scores[
        "feature_timestamp"
    ].max()

    high_count = int(
        (scores["risk_level"] == "HIGH").sum()
    )

    medium_count = int(
        (scores["risk_level"] == "MEDIUM").sum()
    )

    low_count = int(
        (scores["risk_level"] == "LOW").sum()
    )

    top_rows = scores.sort_values(
        "risk_score",
        ascending=False,
    ).head(20)

    print("=" * 70)
    print("ML6 - BATCH SCORE ALL GRIDS")
    print("=" * 70)

    print(f"Database: {DB_PATH}")
    print(f"Model: {artifact['model_name']}")
    print(f"Model version: {artifact['model_version']}")
    print(f"Scored grids: {len(scores):,}")
    print(f"Feature timestamp: {latest_timestamp}")

    print("\nRisk-level distribution:")
    print(f"  HIGH:   {high_count:,}")
    print(f"  MEDIUM: {medium_count:,}")
    print(f"  LOW:    {low_count:,}")

    print("\nTop 20 operational attention:")
    print(
        top_rows[
            [
                "grid_id",
                "feature_timestamp",
                "risk_score",
                "risk_level",
            ]
        ].to_string(index=False)
    )

    published_count = connection.execute(
        "SELECT COUNT(*) FROM network_risk_scores"
    ).fetchone()[0]

    print(
        f"\nPublished network_risk_scores rows: "
        f"{published_count:,}"
    )

    print("\nML6 validation: PASSED")
    print("=" * 70)


def main() -> None:
    print("=" * 70)
    print("ML6 - BATCH SCORE ALL GRIDS")
    print("=" * 70)

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Warehouse database not found: {DB_PATH}"
        )

    artifact = load_model_artifact()

    connection = sqlite3.connect(DB_PATH)

    try:
        print("\n[1] Validating feature table...")
        validate_feature_table(connection)
        print("    Feature table: READY")

        print("\n[2] Reading latest feature row per grid...")
        features = read_latest_features(connection)
        print(
            f"    Latest feature rows: {len(features):,}"
        )

        print("\n[3] Validating feature input...")
        validate_input_features(features)
        print("    Feature validation: PASSED")

        print("\n[4] Running batch model scoring...")
        scores = score_batch(
            features,
            artifact,
        )
        print(
            f"    Scored rows: {len(scores):,}"
        )

        print("\n[5] Creating risk-score table...")
        create_risk_table(connection)
        print("    network_risk_scores: READY")

        print("\n[6] Publishing scores...")
        publish_scores(
            connection,
            scores,
        )
        print("    Scores published")

        print("\n[7] Validating published scores...")
        validate_published_scores(
            connection,
            expected_count=len(scores),
        )
        print("    Publication validation: PASSED")

        print()
        print_summary(
            connection,
            scores,
            artifact,
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()