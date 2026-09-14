
"""
ML3 - TRAIN AND EVALUATE ANOMALOUS ACTIVITY MODEL

Purpose
-------
Predict whether the NEXT hourly activity interval for a grid will
show unusually high or low communication activity.

Prediction unit:
    grid_id + hourly interval

Feature timestamp:
    t

Prediction target:
    unusual activity at t+1

Important:
    - Features only use information available through t.
    - Chronological train/test split.
    - Thresholds are calculated from TRAINING data only.
    - 7 days of source history are used as a limited-history prototype.
    - This is a proxy anomaly label, not ground-truth anomaly detection.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = (
    PROJECT_ROOT
    / "warehouse"
    / "network_analytics.db"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "artifacts"
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "ml3_logistic_regression.pkl"
)

METRICS_PATH = (
    ARTIFACT_DIR
    / "ml3_metrics.json"
)

MODEL_VERSION = "ml3-logistic-v1"

FEATURE_COLUMNS = [
    "avg_activity",
    "activity_growth",
    "active_hours",
    "peak_ratio",
    "variability",
    "internet_share",
]

REQUIRED_FEATURE_COLUMNS = [
    "grid_id",
    "feature_timestamp",
    *FEATURE_COLUMNS,
]

TRAIN_RATIO = 0.80

HIGH_QUANTILE = 0.95
LOW_QUANTILE = 0.05


# ============================================================================
# HELPERS
# ============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def connect_database() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    return sqlite3.connect(DB_PATH)


def ensure_artifact_directory() -> None:
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def finite_check(
    df: pd.DataFrame,
    columns: list[str],
) -> int:
    """
    Count rows containing non-finite numeric values.
    """

    if df.empty:
        return 0

    values = df[columns].to_numpy(
        dtype=float
    )

    return int(
        np.sum(
            ~np.isfinite(values).any(axis=1)
        )
    )


# ============================================================================
# STEP 1 - VALIDATE ML2 FEATURE TABLE
# ============================================================================

def validate_feature_table(
    connection: sqlite3.Connection,
) -> None:

    print_header(
        "1. VALIDATING ML2 FEATURE TABLE"
    )

    tables = pd.read_sql_query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """,
        connection,
    )

    available_tables = set(
        tables["name"].tolist()
    )

    if "network_features" not in available_tables:
        raise RuntimeError(
            "network_features table not found."
        )

    print(
        "network_features table: FOUND"
    )

    schema = pd.read_sql_query(
        "PRAGMA table_info(network_features)",
        connection,
    )

    available_columns = set(
        schema["name"].tolist()
    )

    print("Required columns:")

    for column in REQUIRED_FEATURE_COLUMNS:

        if column in available_columns:

            print(
                f"  {column}: OK"
            )

        else:

            raise RuntimeError(
                f"Missing required feature column: "
                f"{column}"
            )


# ============================================================================
# STEP 2 - LOAD ML2 FEATURES
# ============================================================================

def load_features(
    connection: sqlite3.Connection,
) -> pd.DataFrame:

    print_header(
        "2. LOADING ML2 FEATURES"
    )

    columns_sql = ", ".join(
        f'"{column}"'
        for column in REQUIRED_FEATURE_COLUMNS
    )

    query = f"""
        SELECT
            {columns_sql}
        FROM network_features
        ORDER BY
            feature_timestamp,
            grid_id
    """

    df = pd.read_sql_query(
        query,
        connection,
    )

    if df.empty:
        raise RuntimeError(
            "network_features returned zero rows."
        )

    df["feature_timestamp"] = pd.to_datetime(
        df["feature_timestamp"],
        errors="coerce",
    )

    if df["feature_timestamp"].isna().any():
        raise RuntimeError(
            "Invalid feature_timestamp values detected."
        )

    for column in FEATURE_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    null_counts = df[
        REQUIRED_FEATURE_COLUMNS
    ].isna().sum()

    bad_nulls = null_counts[
        null_counts > 0
    ]

    if not bad_nulls.empty:

        print(
            "Null values detected:"
        )

        print(
            bad_nulls
        )

        raise RuntimeError(
            "ML2 feature table contains "
            "required null values."
        )

    bad_numeric = finite_check(
        df,
        FEATURE_COLUMNS,
    )

    if bad_numeric > 0:

        raise RuntimeError(
            f"Found {bad_numeric} rows with "
            "non-finite feature values."
        )

    print(
        f"Feature rows: {len(df):,}"
    )

    print(
        "Feature timestamp range: "
        f"{df['feature_timestamp'].min()} "
        f"to "
        f"{df['feature_timestamp'].max()}"
    )

    print(
        f"Grids: "
        f"{df['grid_id'].nunique():,}"
    )

    print(
        f"Feature timestamps: "
        f"{df['feature_timestamp'].nunique():,}"
    )

    return df


# ============================================================================
# STEP 3 - LOAD HOURLY ACTIVITY
# ============================================================================

def load_hourly_activity(
    connection: sqlite3.Connection,
) -> pd.DataFrame:

    print_header(
        "3. LOADING HOURLY ACTIVITY"
    )

    """
    Actual warehouse schema:

    fact_network_activity:
        time_id
        grid_id
        sms_in
        sms_out
        call_in
        call_out
        internet_activity
        total_activity

    dim_time:
        time_id
        timestamp
        date
        hour
        day_of_week

    Correct join:

        f.time_id = t.time_id
    """

    query = """
        SELECT
            f.grid_id,
            t.timestamp,
            f.sms_in,
            f.sms_out,
            f.call_in,
            f.call_out,
            f.internet_activity,
            f.total_activity
        FROM fact_network_activity f
        INNER JOIN dim_time t
            ON f.time_id = t.time_id
        ORDER BY
            f.grid_id,
            t.timestamp
    """

    df = pd.read_sql_query(
        query,
        connection,
    )

    if df.empty:
        raise RuntimeError(
            "fact_network_activity returned zero rows."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():

        raise RuntimeError(
            "Invalid timestamps found "
            "in hourly activity."
        )

    numeric_columns = [
        "sms_in",
        "sms_out",
        "call_in",
        "call_out",
        "internet_activity",
        "total_activity",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df["total_activity"].isna().any():

        raise RuntimeError(
            "total_activity contains null values."
        )

    df = (
        df
        .sort_values(
            ["grid_id", "timestamp"]
        )
        .reset_index(drop=True)
    )

    print(
        f"Hourly activity rows: "
        f"{len(df):,}"
    )

    print(
        "Activity timestamp range: "
        f"{df['timestamp'].min()} "
        f"to "
        f"{df['timestamp'].max()}"
    )

    print(
        f"Grids: "
        f"{df['grid_id'].nunique():,}"
    )

    print(
        f"Hourly timestamps: "
        f"{df['timestamp'].nunique():,}"
    )

    return df


# ============================================================================
# STEP 4 - BUILD t -> t+1 TARGET DATASET
# ============================================================================

def build_prediction_dataset(
    features: pd.DataFrame,
    activity: pd.DataFrame,
) -> pd.DataFrame:

    print_header(
        "4. BUILDING NEXT-HOUR PREDICTION DATASET"
    )

    """
    For every feature row at time t:

        features(t) -> predict activity(t+1)

    The next observation is obtained separately
    from the activity table.

    This prevents t+1 from being used as an
    input feature.
    """

    future_activity = activity[
        [
            "grid_id",
            "timestamp",
            "total_activity",
        ]
    ].copy()

    future_activity[
        "feature_timestamp"
    ] = (
        future_activity["timestamp"]
        - pd.Timedelta(hours=1)
    )

    future_activity = future_activity.rename(
        columns={
            "total_activity":
                "next_total_activity"
        }
    )

    future_activity = future_activity[
        [
            "grid_id",
            "feature_timestamp",
            "next_total_activity",
        ]
    ]

    dataset = features.merge(
        future_activity,
        on=[
            "grid_id",
            "feature_timestamp",
        ],
        how="inner",
    )

    dataset = (
        dataset
        .sort_values(
            [
                "feature_timestamp",
                "grid_id",
            ]
        )
        .reset_index(drop=True)
    )

    if dataset.empty:

        raise RuntimeError(
            "No feature rows could be matched "
            "with a next-hour observation."
        )

    print(
        f"Prediction rows with t+1 available: "
        f"{len(dataset):,}"
    )

    print(
        "Prediction timestamp range: "
        f"{dataset['feature_timestamp'].min()} "
        f"to "
        f"{dataset['feature_timestamp'].max()}"
    )

    return dataset


# ============================================================================
# STEP 5 - CHRONOLOGICAL SPLIT
# ============================================================================

def chronological_split(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print_header(
        "5. CHRONOLOGICAL TRAIN / TEST SPLIT"
    )

    timestamps = (
        dataset["feature_timestamp"]
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    split_index = int(
        len(timestamps) * TRAIN_RATIO
    )

    if split_index <= 0:

        raise RuntimeError(
            "Training period is empty."
        )

    if split_index >= len(timestamps):

        raise RuntimeError(
            "Test period is empty."
        )

    train_end_timestamp = (
        timestamps.iloc[split_index - 1]
    )

    test_start_timestamp = (
        timestamps.iloc[split_index]
    )

    train = dataset[
        dataset["feature_timestamp"]
        <= train_end_timestamp
    ].copy()

    test = dataset[
        dataset["feature_timestamp"]
        >= test_start_timestamp
    ].copy()

    print(
        f"Train rows: {len(train):,}"
    )

    print(
        f"Test rows: {len(test):,}"
    )

    print(
        "Train time range: "
        f"{train['feature_timestamp'].min()} "
        f"to "
        f"{train['feature_timestamp'].max()}"
    )

    print(
        "Test time range: "
        f"{test['feature_timestamp'].min()} "
        f"to "
        f"{test['feature_timestamp'].max()}"
    )

    if (
        train["feature_timestamp"].max()
        >=
        test["feature_timestamp"].min()
    ):

        raise RuntimeError(
            "Chronological split failed."
        )

    print(
        "Chronological ordering: PASSED"
    )

    return train, test


# ============================================================================
# STEP 6 - CREATE PROXY ANOMALY TARGET
# ============================================================================

def create_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    float,
    float,
]:

    print_header(
        "6. CREATING NEXT-HOUR "
        "UNUSUAL-ACTIVITY TARGET"
    )

    """
    Thresholds are calculated ONLY from
    training data.

    high_threshold:
        95th percentile of next-hour activity

    low_threshold:
        5th percentile of next-hour activity

    Target:

        1 = unusually high OR unusually low
            next-hour activity

        0 = normal next-hour activity

    This is a proxy target.
    It is NOT a ground-truth anomaly label.
    """

    high_threshold = float(
        train["next_total_activity"].quantile(
            HIGH_QUANTILE
        )
    )

    low_threshold = float(
        train["next_total_activity"].quantile(
            LOW_QUANTILE
        )
    )

    print(
        "Training high threshold "
        "(95th percentile): "
        f"{high_threshold:.6f}"
    )

    print(
        "Training low threshold "
        "(5th percentile): "
        f"{low_threshold:.6f}"
    )

    if low_threshold > high_threshold:

        raise RuntimeError(
            "Invalid anomaly thresholds."
        )

    def assign_target(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        df["target"] = (
            (
                df["next_total_activity"]
                > high_threshold
            )
            |
            (
                df["next_total_activity"]
                < low_threshold
            )
        ).astype(int)

        return df

    train = assign_target(
        train
    )

    test = assign_target(
        test
    )

    train_rate = float(
        train["target"].mean()
    )

    test_rate = float(
        test["target"].mean()
    )

    print(
        f"Train unusual rate: "
        f"{train_rate:.4%}"
    )

    print(
        f"Test unusual rate: "
        f"{test_rate:.4%}"
    )

    print()
    print(
        "Train target distribution:"
    )

    print(
        train["target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Test target distribution:"
    )

    print(
        test["target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    if train["target"].nunique() < 2:

        raise RuntimeError(
            "Training target contains "
            "only one class."
        )

    if test["target"].nunique() < 2:

        print(
            "WARNING: test target contains "
            "only one class."
        )

    return (
        train,
        test,
        low_threshold,
        high_threshold,
    )


# ============================================================================
# STEP 7 - LEAKAGE VALIDATION
# ============================================================================

def validate_no_target_leakage(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:

    print_header(
        "7. VALIDATING TARGET LEAKAGE"
    )

    """
    next_total_activity is used only to
    construct the target.

    It must NEVER be passed to the model.
    """

    forbidden_columns = {
        "next_total_activity",
        "timestamp",
    }

    model_columns = set(
        FEATURE_COLUMNS
    )

    overlap = model_columns.intersection(
        forbidden_columns
    )

    if overlap:

        raise RuntimeError(
            "Leakage columns found in model "
            f"features: {overlap}"
        )

    print(
        "Future target column excluded "
        "from model inputs: PASSED"
    )

    print(
        "Model input feature count: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        "Leakage validation: PASSED"
    )


# ============================================================================
# STEP 8 - TRAIN MODEL
# ============================================================================

def train_model(
    train: pd.DataFrame,
) -> Pipeline:

    print_header(
        "8. TRAINING LOGISTIC REGRESSION"
    )

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train[
        "target"
    ]

    print(
        f"Training samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Positive samples: "
        f"{int(y_train.sum()):,}"
    )

    print(
        f"Negative samples: "
        f"{int((y_train == 0).sum()):,}"
    )

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Logistic Regression training: PASSED"
    )

    return model


# ============================================================================
# STEP 9 - EVALUATE MODEL
# ============================================================================

def evaluate_model(
    model: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:

    print_header(
        "9. EVALUATING MODEL"
    )

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train[
        "target"
    ]

    X_test = test[
        FEATURE_COLUMNS
    ]

    y_test = test[
        "target"
    ]

    train_predictions = model.predict(
        X_train
    )

    test_predictions = model.predict(
        X_test
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    train_accuracy = accuracy_score(
        y_train,
        train_predictions,
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    print(
        f"Training accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test precision: "
        f"{test_precision:.4f}"
    )

    print(
        f"Test recall: "
        f"{test_recall:.4f}"
    )

    print(
        f"Test F1: "
        f"{test_f1:.4f}"
    )

    if y_test.nunique() >= 2:

        test_auc = roc_auc_score(
            y_test,
            test_probabilities,
        )

        print(
            f"Test ROC-AUC: "
            f"{test_auc:.4f}"
        )

    else:

        test_auc = None

        print(
            "Test ROC-AUC: N/A "
            "(only one class in test set)"
        )

    cm = confusion_matrix(
        y_test,
        test_predictions,
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        cm
    )

    print()
    print(
        "Test unusual-activity base rate: "
        f"{float(y_test.mean()):.4%}"
    )

    print()
    print(
        "NOTE:"
    )

    print(
        "Accuracy alone is not sufficient "
        "because unusual activity may be "
        "much less common than normal activity."
    )

    metrics = {
        "model_version": MODEL_VERSION,
        "model_type": "Logistic Regression",
        "prediction_target":
            "unusual_activity_at_t_plus_1",
        "train_accuracy":
            float(train_accuracy),
        "test_accuracy":
            float(test_accuracy),
        "test_precision":
            float(test_precision),
        "test_recall":
            float(test_recall),
        "test_f1":
            float(test_f1),
        "test_roc_auc":
            (
                None
                if test_auc is None
                else float(test_auc)
            ),
        "test_base_rate":
            float(y_test.mean()),
        "confusion_matrix":
            cm.tolist(),
        "train_rows":
            int(len(train)),
        "test_rows":
            int(len(test)),
    }

    return metrics


# ============================================================================
# STEP 10 - FEATURE COEFFICIENTS
# ============================================================================

def show_feature_coefficients(
    model: Pipeline,
) -> list[dict]:

    print_header(
        "10. FEATURE COEFFICIENTS"
    )

    classifier = model.named_steps[
        "classifier"
    ]

    coefficients = (
        classifier.coef_[0]
    )

    rows = []

    for feature, coefficient in zip(
        FEATURE_COLUMNS,
        coefficients,
    ):

        rows.append(
            {
                "feature":
                    feature,
                "coefficient":
                    float(coefficient),
                "absolute_coefficient":
                    float(
                        abs(coefficient)
                    ),
            }
        )

    rows.sort(
        key=lambda x:
            x["absolute_coefficient"],
        reverse=True,
    )

    for row in rows:

        direction = (
            "increases"
            if row["coefficient"] > 0
            else "decreases"
        )

        print(
            f"{row['feature']:20s} "
            f"{row['coefficient']: .6f} "
            f"({direction} unusual-activity probability)"
        )

    return rows


# ============================================================================
# STEP 11 - SIMPLE THRESHOLD BASELINE
# ============================================================================

def evaluate_threshold_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    low_threshold: float,
    high_threshold: float,
) -> dict:

    print_header(
        "11. SIMPLE THRESHOLD BASELINE"
    )

    """
    Baseline:

        Predict unusual when CURRENT
        avg_activity is outside the same
        TRAINING-derived thresholds.

    The thresholds are calculated from
    training data only.
    """

    baseline_predictions = (
        (
            test["avg_activity"]
            > high_threshold
        )
        |
        (
            test["avg_activity"]
            < low_threshold
        )
    ).astype(int)

    y_test = test[
        "target"
    ]

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions,
    )

    baseline_precision = precision_score(
        y_test,
        baseline_predictions,
        zero_division=0,
    )

    baseline_recall = recall_score(
        y_test,
        baseline_predictions,
        zero_division=0,
    )

    baseline_f1 = f1_score(
        y_test,
        baseline_predictions,
        zero_division=0,
    )

    print(
        f"Baseline accuracy: "
        f"{baseline_accuracy:.4f}"
    )

    print(
        f"Baseline precision: "
        f"{baseline_precision:.4f}"
    )

    print(
        f"Baseline recall: "
        f"{baseline_recall:.4f}"
    )

    print(
        f"Baseline F1: "
        f"{baseline_f1:.4f}"
    )

    return {
        "baseline_type":
            "train-derived activity threshold",
        "accuracy":
            float(baseline_accuracy),
        "precision":
            float(baseline_precision),
        "recall":
            float(baseline_recall),
        "f1":
            float(baseline_f1),
    }


# ============================================================================
# STEP 12 - HAND CHECK GRID 4821
# ============================================================================

def hand_check_grid_4821(
    dataset: pd.DataFrame,
) -> None:

    print_header(
        "12. HAND CHECK - GRID 4821"
    )

    grid = dataset[
        dataset["grid_id"] == 4821
    ].sort_values(
        "feature_timestamp"
    )

    if grid.empty:

        print(
            "Grid 4821 not found "
            "in prediction dataset."
        )

        return

    latest = grid.iloc[-1]

    print(
        f"Grid: "
        f"{int(latest['grid_id'])}"
    )

    print(
        f"Feature timestamp: "
        f"{latest['feature_timestamp']}"
    )

    print(
        f"avg_activity: "
        f"{latest['avg_activity']:.6f}"
    )

    print(
        f"activity_growth: "
        f"{latest['activity_growth']:.6f}"
    )

    print(
        f"active_hours: "
        f"{latest['active_hours']:.0f}"
    )

    print(
        f"peak_ratio: "
        f"{latest['peak_ratio']:.6f}"
    )

    print(
        f"variability: "
        f"{latest['variability']:.6f}"
    )

    print(
        f"internet_share: "
        f"{latest['internet_share']:.6f}"
    )

    print(
        f"next_total_activity: "
        f"{latest['next_total_activity']:.6f}"
    )

    if "target" in latest.index:

        print(
            f"target: "
            f"{int(latest['target'])}"
        )

    print(
        "Grid 4821 hand check: COMPLETED"
    )


# ============================================================================
# STEP 13 - SAVE MODEL AND METRICS
# ============================================================================

def save_model(
    model: Pipeline,
    metrics: dict,
    feature_coefficients: list[dict],
    low_threshold: float,
    high_threshold: float,
    baseline_metrics: dict,
) -> None:

    print_header(
        "13. SAVING ML3 ARTIFACTS"
    )

    ensure_artifact_directory()

    artifact = {
        "model": model,
        "model_version":
            MODEL_VERSION,
        "feature_columns":
            FEATURE_COLUMNS,
        "target_definition":
            (
                "1 if next-hour activity is below "
                "the training 5th percentile or "
                "above the training 95th percentile; "
                "otherwise 0"
            ),
        "low_threshold":
            low_threshold,
        "high_threshold":
            high_threshold,
    }

    with open(
        MODEL_PATH,
        "wb",
    ) as file:

        pickle.dump(
            artifact,
            file,
        )

    metrics_output = {
        **metrics,

        "feature_coefficients":
            feature_coefficients,

        "thresholds":
            {
                "low_5_percentile":
                    low_threshold,
                "high_95_percentile":
                    high_threshold,
            },

        "baseline":
            baseline_metrics,

        "history_limit":
            "7-day source history",

        "methodology_note":
            (
                "Prototype trained on 7 days "
                "of source data. Target is a "
                "proxy unusual-activity label, "
                "not ground-truth anomaly detection."
            ),
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics_output,
            file,
            indent=2,
        )

    print(
        f"Model saved: "
        f"{MODEL_PATH}"
    )

    print(
        f"Metrics saved: "
        f"{METRICS_PATH}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print_header(
        "ML3 - TRAIN AND EVALUATE "
        "ANOMALOUS ACTIVITY MODEL"
    )

    print(
        f"Database: {DB_PATH}"
    )

    print(
        "Available history: 7 days"
    )

    print(
        "Model: Logistic Regression"
    )

    print(
        "Split: Chronological 80/20"
    )

    print(
        "Target: Unusual activity at t+1"
    )

    connection = connect_database()

    try:

        # --------------------------------------------------------------
        # 1. Validate ML2 feature table
        # --------------------------------------------------------------

        validate_feature_table(
            connection
        )

        # --------------------------------------------------------------
        # 2. Load ML2 features
        # --------------------------------------------------------------

        features = load_features(
            connection
        )

        # --------------------------------------------------------------
        # 3. Load hourly activity
        # --------------------------------------------------------------

        activity = load_hourly_activity(
            connection
        )

        # --------------------------------------------------------------
        # 4. Build t -> t+1 dataset
        # --------------------------------------------------------------

        dataset = build_prediction_dataset(
            features,
            activity,
        )

        # --------------------------------------------------------------
        # 5. Chronological split
        # --------------------------------------------------------------

        train, test = chronological_split(
            dataset
        )

        # --------------------------------------------------------------
        # 6. Create target
        # --------------------------------------------------------------

        (
            train,
            test,
            low_threshold,
            high_threshold,
        ) = create_target(
            train,
            test,
        )

        # --------------------------------------------------------------
        # 7. Leakage validation
        # --------------------------------------------------------------

        validate_no_target_leakage(
            train,
            test,
        )

        # --------------------------------------------------------------
        # 8. Train model
        # --------------------------------------------------------------

        model = train_model(
            train
        )

        # --------------------------------------------------------------
        # 9. Evaluate model
        # --------------------------------------------------------------

        metrics = evaluate_model(
            model,
            train,
            test,
        )

        # --------------------------------------------------------------
        # 10. Feature coefficients
        # --------------------------------------------------------------

        feature_coefficients = (
            show_feature_coefficients(
                model
            )
        )

        # --------------------------------------------------------------
        # 11. Baseline
        # --------------------------------------------------------------

        baseline_metrics = (
            evaluate_threshold_baseline(
                train,
                test,
                low_threshold,
                high_threshold,
            )
        )

        # --------------------------------------------------------------
        # 12. Hand check
        #
        # IMPORTANT:
        # Use the target-enriched train/test
        # dataset, NOT the original dataset.
        # --------------------------------------------------------------

        hand_check_dataset = pd.concat(
            [
                train,
                test,
            ],
            ignore_index=True,
        )

        hand_check_grid_4821(
            hand_check_dataset
        )

        # --------------------------------------------------------------
        # 13. Save artifacts
        # --------------------------------------------------------------

        save_model(
            model,
            metrics,
            feature_coefficients,
            low_threshold,
            high_threshold,
            baseline_metrics,
        )

        # --------------------------------------------------------------
        # FINAL SUMMARY
        # --------------------------------------------------------------

        print_header(
            "ML3 COMPLETE"
        )

        print(
            f"Model version: "
            f"{MODEL_VERSION}"
        )

        print(
            f"Train accuracy: "
            f"{metrics['train_accuracy']:.4f}"
        )

        print(
            f"Test accuracy: "
            f"{metrics['test_accuracy']:.4f}"
        )

        print(
            f"Test precision: "
            f"{metrics['test_precision']:.4f}"
        )

        print(
            f"Test recall: "
            f"{metrics['test_recall']:.4f}"
        )

        print(
            f"Test F1: "
            f"{metrics['test_f1']:.4f}"
        )

        if metrics["test_roc_auc"] is not None:

            print(
                f"Test ROC-AUC: "
                f"{metrics['test_roc_auc']:.4f}"
            )

        print()

        print(
            f"Baseline F1: "
            f"{baseline_metrics['f1']:.4f}"
        )

        print()

        print(
            "ML3 STATUS: PASSED"
        )

    finally:

        connection.close()


if __name__ == "__main__":
    main()

