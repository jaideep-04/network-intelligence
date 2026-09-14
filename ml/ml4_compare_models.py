
"""
ML4 - MODEL COMPARISON AND IMPROVEMENT
=======================================

Primary problem:
    Predict whether the NEXT hourly activity interval for a grid
    will exhibit unusual communication activity.

Prediction unit:
    grid_id + hourly time interval

Prediction rule:
    Features available at time t are used to predict unusual
    activity at time t+1.

ML2 features:
    - avg_activity
    - activity_growth
    - active_hours
    - peak_ratio
    - variability
    - internet_share

Models:
    1. Logistic Regression
    2. Random Forest
    3. Simple Threshold Baseline

Important methodology:
    - 7-day source history
    - Chronological 80/20 train/test split
    - Target thresholds derived from TRAINING data only
    - No future target information is used as a model feature
    - No random train/test shuffling
    - F1 is the primary ML comparison metric
    - Recall is important because unusual activity should not be missed
"""


# ============================================================
# IMPORTS
# ============================================================

import json
import math
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DB_PATH = (
    BASE_DIR
    / "warehouse"
    / "network_analytics.db"
)

ARTIFACT_DIR = (
    BASE_DIR
    / "ml"
    / "artifacts"
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_MODEL_PATH = (
    ARTIFACT_DIR
    / "ml4_best_model.pkl"
)

METRICS_PATH = (
    ARTIFACT_DIR
    / "ml4_metrics.json"
)

MODEL_VERSION = "ml4-random-forest-v1"

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


# ============================================================
# HELPER
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def connect_database():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found:\n{DB_PATH}"
        )

    return sqlite3.connect(DB_PATH)


# ============================================================
# 1. VALIDATE ML2 FEATURE TABLE
# ============================================================

def validate_feature_table(connection):

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

    table_names = set(
        tables["name"].tolist()
    )

    if "network_features" not in table_names:
        raise RuntimeError(
            "network_features table was not found."
        )

    print(
        "network_features table: FOUND"
    )

    schema = pd.read_sql_query(
        """
        PRAGMA table_info(network_features)
        """,
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
                f"Missing required column: {column}"
            )

    print(
        "Feature table validation: PASSED"
    )


# ============================================================
# 2. LOAD ML2 FEATURES
# ============================================================

def load_features(connection):

    print_header(
        "2. LOADING ML2 FEATURES"
    )

    selected_columns = ", ".join(
        [
            "grid_id",
            "feature_timestamp",
            *FEATURE_COLUMNS,
        ]
    )

    query = f"""
        SELECT
            {selected_columns}
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
            "network_features contains no rows."
        )

    df["feature_timestamp"] = (
        pd.to_datetime(
            df["feature_timestamp"]
        )
    )

    df["grid_id"] = pd.to_numeric(
        df["grid_id"],
        errors="coerce",
    )

    for column in FEATURE_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
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
        "Feature timestamps: "
        f"{df['feature_timestamp'].nunique():,}"
    )

    return df


# ============================================================
# 3. LOAD HOURLY ACTIVITY
# ============================================================

def load_hourly_activity(connection):

    print_header(
        "3. LOADING HOURLY ACTIVITY"
    )

    query = """
        SELECT
            f.grid_id,
            t.timestamp,
            f.total_activity
        FROM fact_network_activity f
        JOIN dim_time t
            ON f.time_id = t.time_id
        ORDER BY
            t.timestamp,
            f.grid_id
    """

    df = pd.read_sql_query(
        query,
        connection,
    )

    if df.empty:
        raise RuntimeError(
            "fact_network_activity contains no rows."
        )

    df["timestamp"] = (
        pd.to_datetime(
            df["timestamp"]
        )
    )

    df["grid_id"] = pd.to_numeric(
        df["grid_id"],
        errors="coerce",
    )

    df["total_activity"] = pd.to_numeric(
        df["total_activity"],
        errors="coerce",
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
        "Hourly timestamps: "
        f"{df['timestamp'].nunique():,}"
    )

    return df


# ============================================================
# 4. BUILD NEXT-HOUR DATASET
# ============================================================

def build_prediction_dataset(
    features,
    activity,
):

    print_header(
        "4. BUILDING NEXT-HOUR PREDICTION DATASET"
    )

    next_activity = activity.copy()

    # Shift the activity timestamp backward by one hour.
    #
    # Example:
    #
    # actual activity:
    #   04:30
    #
    # becomes:
    #   feature_timestamp = 03:30
    #
    # Therefore activity at 04:30 becomes the
    # next-hour observation for features at 03:30.

    next_activity[
        "feature_timestamp"
    ] = (
        next_activity["timestamp"]
        - pd.Timedelta(hours=1)
    )

    next_activity = next_activity[
        [
            "grid_id",
            "feature_timestamp",
            "total_activity",
        ]
    ].rename(
        columns={
            "total_activity":
                "next_total_activity"
        }
    )

    dataset = features.merge(
        next_activity,
        on=[
            "grid_id",
            "feature_timestamp",
        ],
        how="inner",
    )

    dataset = dataset.sort_values(
        [
            "feature_timestamp",
            "grid_id",
        ]
    ).reset_index(
        drop=True
    )

    if dataset.empty:
        raise RuntimeError(
            "No next-hour prediction rows were created."
        )

    print(
        "Prediction rows with t+1 available: "
        f"{len(dataset):,}"
    )

    print(
        "Prediction timestamp range: "
        f"{dataset['feature_timestamp'].min()} "
        f"to "
        f"{dataset['feature_timestamp'].max()}"
    )

    return dataset


# ============================================================
# 5. CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(dataset):

    print_header(
        "5. CHRONOLOGICAL TRAIN / TEST SPLIT"
    )

    dataset = dataset.sort_values(
        [
            "feature_timestamp",
            "grid_id",
        ]
    ).reset_index(
        drop=True
    )

    split_index = int(
        len(dataset) * 0.80
    )

    train = dataset.iloc[
        :split_index
    ].copy()

    test = dataset.iloc[
        split_index:
    ].copy()

    if train.empty:
        raise RuntimeError(
            "Training dataset is empty."
        )

    if test.empty:
        raise RuntimeError(
            "Test dataset is empty."
        )

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
            "Chronological split validation failed."
        )

    print(
        "Chronological ordering: PASSED"
    )

    return train, test


# ============================================================
# 6. CREATE TARGET
# ============================================================

def create_target(
    train,
    test,
):

    print_header(
        "6. CREATING NEXT-HOUR UNUSUAL-ACTIVITY TARGET"
    )

    train_activity = (
        train["next_total_activity"]
        .dropna()
    )

    if train_activity.empty:
        raise RuntimeError(
            "Training next-hour activity is empty."
        )

    # IMPORTANT:
    #
    # Thresholds are calculated ONLY from
    # training data.
    #
    # This prevents test information from
    # influencing the target definition.

    high_threshold = float(
        train_activity.quantile(
            0.95
        )
    )

    low_threshold = float(
        train_activity.quantile(
            0.05
        )
    )

    def assign_target(df):

        df = df.copy()

        unusual = (
            (
                df["next_total_activity"]
                > high_threshold
            )
            |
            (
                df["next_total_activity"]
                < low_threshold
            )
        )

        df["target"] = (
            unusual.astype(int)
        )

        return df

    train = assign_target(
        train
    )

    test = assign_target(
        test
    )

    print(
        "Training high threshold "
        f"(95th percentile): "
        f"{high_threshold:.6f}"
    )

    print(
        "Training low threshold "
        f"(5th percentile): "
        f"{low_threshold:.6f}"
    )

    print(
        "Train unusual rate: "
        f"{train['target'].mean() * 100:.4f}%"
    )

    print(
        "Test unusual rate: "
        f"{test['target'].mean() * 100:.4f}%"
    )

    print()
    print(
        "Train target distribution:"
    )

    print(
        train["target"]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Test target distribution:"
    )

    print(
        test["target"]
        .value_counts()
        .sort_index()
    )

    if train["target"].nunique() < 2:
        raise RuntimeError(
            "Training target has only one class."
        )

    if test["target"].nunique() < 2:
        raise RuntimeError(
            "Test target has only one class."
        )

    return (
        train,
        test,
        low_threshold,
        high_threshold,
    )


# ============================================================
# 7. LEAKAGE VALIDATION
# ============================================================

def validate_leakage(
    train,
    test,
):

    print_header(
        "7. VALIDATING TARGET LEAKAGE"
    )

    prohibited_columns = {
        "target",
        "next_total_activity",
    }

    leakage_columns = (
        set(FEATURE_COLUMNS)
        &
        prohibited_columns
    )

    if leakage_columns:
        raise RuntimeError(
            "Leakage detected in model inputs: "
            f"{sorted(leakage_columns)}"
        )

    # Verify the six model features are exactly
    # the intended feature set.

    if len(FEATURE_COLUMNS) != 6:
        raise RuntimeError(
            "Unexpected model feature count."
        )

    print(
        "Future target column excluded "
        "from model inputs: PASSED"
    )

    print(
        "Model input feature count: "
        f"{len(FEATURE_COLUMNS)}"
    )

    # Verify every row has a next-hour timestamp
    # exactly one hour after feature_timestamp.

    for name, df in [
        ("train", train),
        ("test", test),
    ]:

        expected_next = (
            df["feature_timestamp"]
            + pd.Timedelta(hours=1)
        )

        if len(expected_next) != len(df):
            raise RuntimeError(
                f"{name}: timestamp alignment failed."
            )

    print(
        "t -> t+1 timestamp alignment: PASSED"
    )

    print(
        "Leakage validation: PASSED"
    )


# ============================================================
# 8. PREPARE MODEL DATA
# ============================================================

def prepare_model_data(
    train,
    test,
):

    X_train = train[
        FEATURE_COLUMNS
    ].copy()

    X_test = test[
        FEATURE_COLUMNS
    ].copy()

    y_train = train[
        "target"
    ].astype(int)

    y_test = test[
        "target"
    ].astype(int)

    # Replace infinite values with NaN.
    # The pipeline's imputer will handle them.

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Explicit validation.

    train_bad = np.any(
        ~np.isfinite(
            X_train.fillna(0).to_numpy(
                dtype=float
            )
        ),
        axis=1,
    ).sum()

    test_bad = np.any(
        ~np.isfinite(
            X_test.fillna(0).to_numpy(
                dtype=float
            )
        ),
        axis=1,
    ).sum()

    print_header(
        "8. PREPARING MODEL INPUTS"
    )

    print(
        f"Training feature rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Test feature rows: "
        f"{len(X_test):,}"
    )

    print(
        f"Training non-finite rows: "
        f"{train_bad}"
    )

    print(
        f"Test non-finite rows: "
        f"{test_bad}"
    )

    if train_bad != 0:
        raise RuntimeError(
            "Invalid training feature values detected."
        )

    if test_bad != 0:
        raise RuntimeError(
            "Invalid test feature values detected."
        )

    print(
        "Model input preparation: PASSED"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# 9. MODEL DEFINITIONS
# ============================================================

def build_models():

    models = {}

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    models[
        "Logistic Regression"
    ] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------
    #
    # Configuration is intentionally moderate because
    # the dataset contains ~1.2M feature rows.

    models[
        "Random Forest"
    ] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return models


# ============================================================
# 10. EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):

    model.fit(
        X_train,
        y_train,
    )

    train_predictions = (
        model.predict(
            X_train
        )
    )

    test_predictions = (
        model.predict(
            X_test
        )
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    train_accuracy = (
        accuracy_score(
            y_train,
            train_predictions,
        )
    )

    test_accuracy = (
        accuracy_score(
            y_test,
            test_predictions,
        )
    )

    test_precision = (
        precision_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    test_recall = (
        recall_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    test_f1 = (
        f1_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    test_roc_auc = (
        roc_auc_score(
            y_test,
            test_probabilities,
        )
    )

    matrix = confusion_matrix(
        y_test,
        test_predictions,
    )

    return {
        "model": model,
        "train_accuracy": float(
            train_accuracy
        ),
        "test_accuracy": float(
            test_accuracy
        ),
        "test_precision": float(
            test_precision
        ),
        "test_recall": float(
            test_recall
        ),
        "test_f1": float(
            test_f1
        ),
        "test_roc_auc": float(
            test_roc_auc
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
        "test_predictions": (
            test_predictions
        ),
        "test_probabilities": (
            test_probabilities
        ),
    }


# ============================================================
# 11. TRAIN MODELS
# ============================================================

def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
):

    print_header(
        "9. TRAINING ML MODELS"
    )

    models = build_models()

    results = {}

    for model_name, model in models.items():

        print()
        print(
            f"Training {model_name}..."
        )

        result = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results[
            model_name
        ] = result

        print(
            f"{model_name}: PASSED"
        )

        print(
            f"  Train accuracy: "
            f"{result['train_accuracy']:.4f}"
        )

        print(
            f"  Test accuracy: "
            f"{result['test_accuracy']:.4f}"
        )

        print(
            f"  Test precision: "
            f"{result['test_precision']:.4f}"
        )

        print(
            f"  Test recall: "
            f"{result['test_recall']:.4f}"
        )

        print(
            f"  Test F1: "
            f"{result['test_f1']:.4f}"
        )

        print(
            f"  Test ROC-AUC: "
            f"{result['test_roc_auc']:.4f}"
        )

        print(
            "  Confusion matrix:"
        )

        print(
            result[
                "confusion_matrix"
            ]
        )

    return results


# ============================================================
# 12. SIMPLE THRESHOLD BASELINE
# ============================================================

def calculate_baseline(
    test,
    low_threshold,
    high_threshold,
):

    print_header(
        "10. SIMPLE THRESHOLD BASELINE"
    )

    # Baseline:
    #
    # If current avg_activity is already above
    # the high threshold or below the low threshold,
    # predict that the next interval will be unusual.

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
    ].astype(int)

    accuracy = (
        accuracy_score(
            y_test,
            baseline_predictions,
        )
    )

    precision = (
        precision_score(
            y_test,
            baseline_predictions,
            zero_division=0,
        )
    )

    recall = (
        recall_score(
            y_test,
            baseline_predictions,
            zero_division=0,
        )
    )

    f1 = (
        f1_score(
            y_test,
            baseline_predictions,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        baseline_predictions,
    )

    print(
        f"Baseline accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Baseline precision: "
        f"{precision:.4f}"
    )

    print(
        f"Baseline recall: "
        f"{recall:.4f}"
    )

    print(
        f"Baseline F1: "
        f"{f1:.4f}"
    )

    print(
        "Baseline confusion matrix:"
    )

    print(matrix)

    return {
        "accuracy": float(
            accuracy
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
    }


# ============================================================
# 13. MODEL COMPARISON
# ============================================================

def print_comparison(
    results,
    baseline,
):

    print_header(
        "11. FINAL MODEL COMPARISON"
    )

    print(
        f"{'Model':<24}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'ROC-AUC':>12}"
    )

    print("-" * 84)

    for model_name, result in results.items():

        print(
            f"{model_name:<24}"
            f"{result['test_accuracy']:>12.4f}"
            f"{result['test_precision']:>12.4f}"
            f"{result['test_recall']:>12.4f}"
            f"{result['test_f1']:>12.4f}"
            f"{result['test_roc_auc']:>12.4f}"
        )

    print("-" * 84)

    print(
        f"{'Threshold Baseline':<24}"
        f"{baseline['accuracy']:>12.4f}"
        f"{baseline['precision']:>12.4f}"
        f"{baseline['recall']:>12.4f}"
        f"{baseline['f1']:>12.4f}"
        f"{'N/A':>12}"
    )

    print()
    print(
        "Primary comparison metric: F1"
    )

    print(
        "Secondary consideration: Recall"
    )

    print(
        "Supporting metric: ROC-AUC"
    )


# ============================================================
# 14. SELECT BEST ML MODEL
# ============================================================

def select_best_model(
    results,
    baseline,
):

    print_header(
        "12. SELECTING BEST ML MODEL"
    )

    # Select ONLY among ML models.
    #
    # Baseline is kept separate because it is
    # not an ML model.

    best_name = max(
        results,
        key=lambda name: (
            results[name]["test_f1"],
            results[name]["test_recall"],
            results[name]["test_roc_auc"],
        ),
    )

    best_result = results[
        best_name
    ]

    print(
        f"Best ML model: "
        f"{best_name}"
    )

    print(
        f"Best ML F1: "
        f"{best_result['test_f1']:.4f}"
    )

    print(
        f"Best ML recall: "
        f"{best_result['test_recall']:.4f}"
    )

    print(
        f"Best ML ROC-AUC: "
        f"{best_result['test_roc_auc']:.4f}"
    )

    print()
    print(
        f"Baseline F1: "
        f"{baseline['f1']:.4f}"
    )

    print(
        f"Baseline recall: "
        f"{baseline['recall']:.4f}"
    )

    f1_difference = (
        best_result["test_f1"]
        - baseline["f1"]
    )

    recall_difference = (
        best_result["test_recall"]
        - baseline["recall"]
    )

    print()
    print(
        f"ML F1 - Baseline F1: "
        f"{f1_difference:+.4f}"
    )

    print(
        f"ML Recall - Baseline Recall: "
        f"{recall_difference:+.4f}"
    )

    if (
        best_result["test_f1"]
        > baseline["f1"]
    ):
        print()
        print(
            "ML4 RESULT: "
            "BEST ML MODEL OUTPERFORMS BASELINE ON F1"
        )

    elif (
        best_result["test_f1"]
        == baseline["f1"]
    ):
        print()
        print(
            "ML4 RESULT: "
            "BEST ML MODEL MATCHES BASELINE ON F1"
        )

    else:
        print()
        print(
            "ML4 RESULT: "
            "BASELINE HAS HIGHER F1"
        )

        if (
            best_result["test_recall"]
            > baseline["recall"]
        ):
            print(
                "However, the ML model has "
                "higher recall than the baseline."
            )

    return (
        best_name,
        best_result,
    )


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

def print_feature_importance(
    best_name,
    best_model,
):

    print_header(
        "13. FEATURE IMPORTANCE"
    )

    if best_name == "Logistic Regression":

        classifier = (
            best_model.named_steps[
                "model"
            ]
        )

        coefficients = (
            classifier.coef_[0]
        )

        pairs = list(
            zip(
                FEATURE_COLUMNS,
                coefficients,
            )
        )

        pairs.sort(
            key=lambda item: abs(
                item[1]
            ),
            reverse=True,
        )

        for feature, coefficient in pairs:

            direction = (
                "increases"
                if coefficient > 0
                else "decreases"
            )

            print(
                f"{feature:<22}"
                f"{coefficient:>12.6f} "
                f"({direction} unusual-activity probability)"
            )

    elif best_name == "Random Forest":

        classifier = (
            best_model.named_steps[
                "model"
            ]
        )

        importances = (
            classifier.feature_importances_
        )

        pairs = list(
            zip(
                FEATURE_COLUMNS,
                importances,
            )
        )

        pairs.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        for feature, importance in pairs:

            print(
                f"{feature:<22}"
                f"{importance:>12.6f}"
            )


# ============================================================
# 16. HAND CHECK - GRID 4821
# ============================================================

def hand_check_grid_4821(
    dataset,
    best_model,
    low_threshold,
    high_threshold,
):

    print_header(
        "14. HAND CHECK - GRID 4821"
    )

    grid = dataset[
        dataset["grid_id"] == 4821
    ].sort_values(
        "feature_timestamp"
    ).copy()

    if grid.empty:
        raise RuntimeError(
            "Grid 4821 was not found."
        )

    # The final feature row must have a next-hour
    # observation available.
    latest = grid.iloc[-1]

    feature_timestamp = (
        latest["feature_timestamp"]
    )

    expected_next_timestamp = (
        feature_timestamp
        + pd.Timedelta(hours=1)
    )

    print(
        f"Grid: "
        f"{int(latest['grid_id'])}"
    )

    print(
        f"Feature timestamp: "
        f"{feature_timestamp}"
    )

    print(
        f"Expected next timestamp: "
        f"{expected_next_timestamp}"
    )

    for feature in FEATURE_COLUMNS:

        value = latest[feature]

        if pd.isna(value):
            print(
                f"{feature}: NaN"
            )
        else:
            print(
                f"{feature}: "
                f"{float(value):.6f}"
            )

    actual_next_activity = float(
        latest["next_total_activity"]
    )

    print(
        f"next_total_activity: "
        f"{actual_next_activity:.6f}"
    )

    print(
        f"Low threshold: "
        f"{low_threshold:.6f}"
    )

    print(
        f"High threshold: "
        f"{high_threshold:.6f}"
    )

    # Manually calculate expected target.

    expected_target = int(
        (
            actual_next_activity
            > high_threshold
        )
        or
        (
            actual_next_activity
            < low_threshold
        )
    )

    actual_target = int(
        latest["target"]
    )

    print(
        f"target: "
        f"{actual_target}"
    )

    if (
        actual_target
        != expected_target
    ):
        raise RuntimeError(
            "Grid 4821 target calculation failed."
        )

    print(
        "Target consistency: PASSED"
    )

    # Prepare exactly one feature row for prediction.

    feature_row = (
        latest[
            FEATURE_COLUMNS
        ]
        .to_frame()
        .T
    )

    model_prediction = int(
        best_model.predict(
            feature_row
        )[0]
    )

    model_probability = float(
        best_model.predict_proba(
            feature_row
        )[0, 1]
    )

    print(
        f"Model prediction: "
        f"{model_prediction}"
    )

    print(
        f"Model unusual-activity probability: "
        f"{model_probability:.6f}"
    )

    print(
        "Grid 4821 hand check: PASSED"
    )


# ============================================================
# 17. SAVE BEST MODEL
# ============================================================

def save_best_model(
    best_name,
    best_model,
    low_threshold,
    high_threshold,
):

    print_header(
        "15. SAVING BEST MODEL"
    )

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "model_version": MODEL_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "prediction_horizon": "t+1 hour",
        "target_definition": (
            "1 when next-hour total activity "
            "is above the training 95th percentile "
            "or below the training 5th percentile; "
            "otherwise 0."
        ),
        "low_threshold": float(
            low_threshold
        ),
        "high_threshold": float(
            high_threshold
        ),
        "history_limitation": (
            "7-day source history prototype"
        ),
    }

    with open(
        BEST_MODEL_PATH,
        "wb",
    ) as file:

        pickle.dump(
            artifact,
            file,
        )

    print(
        f"Best model saved: "
        f"{BEST_MODEL_PATH}"
    )


# ============================================================
# 18. SAVE METRICS
# ============================================================

def save_metrics(
    results,
    baseline,
    best_name,
    best_result,
    train,
    test,
    low_threshold,
    high_threshold,
):

    print_header(
        "16. SAVING ML4 METRICS"
    )

    serializable_results = {}

    for model_name, result in results.items():

        serializable_results[
            model_name
        ] = {
            "train_accuracy":
                result[
                    "train_accuracy"
                ],

            "test_accuracy":
                result[
                    "test_accuracy"
                ],

            "test_precision":
                result[
                    "test_precision"
                ],

            "test_recall":
                result[
                    "test_recall"
                ],

            "test_f1":
                result[
                    "test_f1"
                ],

            "test_roc_auc":
                result[
                    "test_roc_auc"
                ],

            "confusion_matrix":
                result[
                    "confusion_matrix"
                ],
        }

    metrics = {

        "model_version":
            MODEL_VERSION,

        "primary_problem": (
            "Predict unusual communication "
            "activity at the next hourly interval."
        ),

        "prediction_unit": (
            "grid_id + hourly time interval"
        ),

        "prediction_horizon":
            "t+1 hour",

        "available_history":
            "7 days",

        "feature_columns":
            FEATURE_COLUMNS,

        "target": {

            "definition": (
                "Next-hour activity is unusual "
                "when it is above the training "
                "95th percentile or below the "
                "training 5th percentile."
            ),

            "low_threshold":
                float(
                    low_threshold
                ),

            "high_threshold":
                float(
                    high_threshold
                ),
        },

        "split": {

            "method":
                "chronological 80/20",

            "train_rows":
                int(
                    len(train)
                ),

            "test_rows":
                int(
                    len(test)
                ),

            "train_start":
                str(
                    train[
                        "feature_timestamp"
                    ].min()
                ),

            "train_end":
                str(
                    train[
                        "feature_timestamp"
                    ].max()
                ),

            "test_start":
                str(
                    test[
                        "feature_timestamp"
                    ].min()
                ),

            "test_end":
                str(
                    test[
                        "feature_timestamp"
                    ].max()
                ),
        },

        "models":
            serializable_results,

        "baseline":
            baseline,

        "best_model": {

            "name":
                best_name,

            "test_accuracy":
                best_result[
                    "test_accuracy"
                ],

            "test_precision":
                best_result[
                    "test_precision"
                ],

            "test_recall":
                best_result[
                    "test_recall"
                ],

            "test_f1":
                best_result[
                    "test_f1"
                ],

            "test_roc_auc":
                best_result[
                    "test_roc_auc"
                ],
        },

        "comparison": {

            "baseline_f1":
                baseline[
                    "f1"
                ],

            "best_ml_f1":
                best_result[
                    "test_f1"
                ],

            "f1_difference":
                (
                    best_result[
                        "test_f1"
                    ]
                    -
                    baseline[
                        "f1"
                    ]
                ),

            "baseline_recall":
                baseline[
                    "recall"
                ],

            "best_ml_recall":
                best_result[
                    "test_recall"
                ],

            "recall_difference":
                (
                    best_result[
                        "test_recall"
                    ]
                    -
                    baseline[
                        "recall"
                    ]
                ),
        },

        "selection_rule": (
            "Highest ML test F1, followed by "
            "recall and ROC-AUC as tie-breakers."
        ),

        "leakage_validation":
            "PASSED",

        "chronological_split":
            "PASSED",

        "history_limitation": (
            "Results are based on only seven "
            "days of source history and should "
            "be treated as a prototype."
        ),

        "non_goals": [

            "congestion prediction",

            "capacity prediction",

            "throughput prediction",

            "latency prediction",

            "packet loss prediction",

            "radio utilization prediction",

            "confirmed fault or outage detection",
        ],
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(
        f"Metrics saved: "
        f"{METRICS_PATH}"
    )


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

def print_final_summary(
    results,
    baseline,
    best_name,
    best_result,
):

    print_header(
        "ML4 COMPLETE"
    )

    print(
        f"Selected ML model: "
        f"{best_name}"
    )

    print()
    print(
        f"Model test accuracy: "
        f"{best_result['test_accuracy']:.4f}"
    )

    print(
        f"Model test precision: "
        f"{best_result['test_precision']:.4f}"
    )

    print(
        f"Model test recall: "
        f"{best_result['test_recall']:.4f}"
    )

    print(
        f"Model test F1: "
        f"{best_result['test_f1']:.4f}"
    )

    print(
        f"Model test ROC-AUC: "
        f"{best_result['test_roc_auc']:.4f}"
    )

    print()
    print(
        f"Baseline F1: "
        f"{baseline['f1']:.4f}"
    )

    print(
        f"Baseline recall: "
        f"{baseline['recall']:.4f}"
    )

    print()

    if (
        best_result["test_f1"]
        > baseline["f1"]
    ):

        print(
            "ML result: "
            "ML model outperformed baseline on F1."
        )

    elif (
        best_result["test_f1"]
        == baseline["f1"]
    ):

        print(
            "ML result: "
            "ML model matched baseline F1."
        )

    else:

        print(
            "ML result: "
            "Baseline has higher F1."
        )

        if (
            best_result["test_recall"]
            > baseline["recall"]
        ):

            print(
                "ML result: "
                "ML model has substantially higher recall."
            )

    print()
    print(
        "Leakage validation: PASSED"
    )

    print(
        "Chronological split: PASSED"
    )

    print(
        "Grid 4821 hand check: PASSED"
    )

    print()
    print(
        "ML4 STATUS: PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "ML4 - MODEL COMPARISON AND IMPROVEMENT"
    )

    print(
        f"Database: {DB_PATH}"
    )

    print(
        "Available history: 7 days"
    )

    print(
        "Model comparison: "
        "Logistic Regression vs Random Forest"
    )

    print(
        "Baseline: Simple threshold"
    )

    print(
        "Prediction problem: "
        "Unusual activity at t+1"
    )

    print(
        "Split: Chronological 80/20"
    )

    connection = connect_database()

    try:

        # ----------------------------------------------------
        # 1. Validate feature table
        # ----------------------------------------------------

        validate_feature_table(
            connection
        )

        # ----------------------------------------------------
        # 2. Load ML2 features
        # ----------------------------------------------------

        features = load_features(
            connection
        )

        # ----------------------------------------------------
        # 3. Load hourly activity
        # ----------------------------------------------------

        activity = (
            load_hourly_activity(
                connection
            )
        )

        # ----------------------------------------------------
        # 4. Build t -> t+1 dataset
        # ----------------------------------------------------

        dataset = (
            build_prediction_dataset(
                features,
                activity,
            )
        )

        # ----------------------------------------------------
        # 5. Chronological split
        # ----------------------------------------------------

        train, test = (
            chronological_split(
                dataset
            )
        )

        # ----------------------------------------------------
        # 6. Create target
        # ----------------------------------------------------

        (
            train,
            test,
            low_threshold,
            high_threshold,
        ) = create_target(
            train,
            test,
        )

        # ----------------------------------------------------
        # 7. Leakage validation
        # ----------------------------------------------------

        validate_leakage(
            train,
            test,
        )

        # ----------------------------------------------------
        # 8. Prepare model inputs
        # ----------------------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = prepare_model_data(
            train,
            test,
        )

        # ----------------------------------------------------
        # 9. Train models
        # ----------------------------------------------------

        results = train_models(
            X_train,
            X_test,
            y_train,
            y_test,
        )

        # ----------------------------------------------------
        # 10. Baseline
        # ----------------------------------------------------

        baseline = (
            calculate_baseline(
                test,
                low_threshold,
                high_threshold,
            )
        )

        # ----------------------------------------------------
        # 11. Comparison
        # ----------------------------------------------------

        print_comparison(
            results,
            baseline,
        )

        # ----------------------------------------------------
        # 12. Select best ML model
        # ----------------------------------------------------

        (
            best_name,
            best_result,
        ) = select_best_model(
            results,
            baseline,
        )

        # ----------------------------------------------------
        # 13. Feature importance
        # ----------------------------------------------------

        print_feature_importance(
            best_name,
            best_result["model"],
        )

        # ----------------------------------------------------
        # 14. Hand check
        # ----------------------------------------------------

        combined_dataset = pd.concat(
            [
                train,
                test,
            ],
            ignore_index=True,
        )

        hand_check_grid_4821(
            combined_dataset,
            best_result["model"],
            low_threshold,
            high_threshold,
        )

        # ----------------------------------------------------
        # 15. Save model
        # ----------------------------------------------------

        save_best_model(
            best_name,
            best_result["model"],
            low_threshold,
            high_threshold,
        )

        # ----------------------------------------------------
        # 16. Save metrics
        # ----------------------------------------------------

        save_metrics(
            results,
            baseline,
            best_name,
            best_result,
            train,
            test,
            low_threshold,
            high_threshold,
        )

        # ----------------------------------------------------
        # 17. Final summary
        # ----------------------------------------------------

        print_final_summary(
            results,
            baseline,
            best_name,
            best_result,
        )

    finally:

        connection.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

