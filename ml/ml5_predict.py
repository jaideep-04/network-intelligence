from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# ML5 - MODEL PREDICTION SERVICE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "artifacts"
    / "ml4_best_model.pkl"
)


# ============================================================
# MODEL FEATURE CONTRACT
# ============================================================

EXPECTED_FEATURES = [
    "avg_activity",
    "activity_growth",
    "active_hours",
    "peak_ratio",
    "variability",
    "internet_share",
]


# ============================================================
# LOAD MODEL ARTIFACT
# ============================================================

def load_model_artifact():
    """
    Load the complete ML4 model artifact.

    The ML4 artifact contains:
        model
        model_name
        model_version
        feature_columns
        prediction_horizon
        target_definition
        low_threshold
        high_threshold
        history_limitation
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ML model artifact not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict):
        raise ValueError(
            "ML model artifact must be a dictionary"
        )

    required_keys = [
        "model",
        "model_name",
        "model_version",
        "feature_columns",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in artifact
    ]

    if missing_keys:
        raise ValueError(
            "ML model artifact is missing required keys: "
            + ", ".join(missing_keys)
        )

    model = artifact["model"]

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The trained model does not support predict_proba()"
        )

    artifact_features = artifact["feature_columns"]

    if artifact_features != EXPECTED_FEATURES:
        raise ValueError(
            "ML model feature columns do not match the "
            "ML5 feature contract.\n"
            f"Expected: {EXPECTED_FEATURES}\n"
            f"Found: {artifact_features}"
        )

    return artifact


# ============================================================
# PREDICT RISK
# ============================================================

def predict_risk(
    avg_activity: float,
    activity_growth: float,
    active_hours: int,
    peak_ratio: float,
    variability: float,
    internet_share: float,
):
    """
    Generate an unusual-activity risk prediction.

    The model predicts:

        0 = normal activity
        1 = unusual activity
    """

    artifact = load_model_artifact()

    model = artifact["model"]

    model_version = str(
        artifact["model_version"]
    )

    feature_columns = artifact["feature_columns"]

    # --------------------------------------------------------
    # Build one-row DataFrame
    # --------------------------------------------------------

    model_input = pd.DataFrame(
        [
            {
                "avg_activity": avg_activity,
                "activity_growth": activity_growth,
                "active_hours": active_hours,
                "peak_ratio": peak_ratio,
                "variability": variability,
                "internet_share": internet_share,
            }
        ],
        columns=feature_columns,
    )

    # --------------------------------------------------------
    # Validate values
    # --------------------------------------------------------

    if model_input.isnull().any().any():
        raise ValueError(
            "Prediction features contain null values"
        )

    if not model_input.apply(
        lambda column: column.map(
            lambda value: isinstance(
                value,
                (int, float),
            )
        )
    ).all().all():
        raise ValueError(
            "Prediction features contain invalid numeric values"
        )

    if not model_input.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    ).notna().all().all():
        raise ValueError(
            "Prediction features contain non-finite values"
        )

    # --------------------------------------------------------
    # Predict probability of class 1
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        model_input
    )

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "ML model does not contain class 1"
        )

    positive_class_index = classes.index(1)

    risk_score = float(
        probabilities[
            0
        ][
            positive_class_index
        ]
    )

    # --------------------------------------------------------
    # Convert score to API risk level
    # --------------------------------------------------------

    if risk_score >= 0.66:
        risk_level = "HIGH"

    elif risk_score >= 0.33:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    return {
        "risk_score": round(
            risk_score,
            6,
        ),
        "risk_level": risk_level,
        "model_version": model_version,
        "explanation_note": (
            "Risk score estimates the likelihood of an "
            "unusual activity pattern in the next hourly "
            "interval. Higher scores indicate that the "
            "grid activity pattern should be investigated. "
            "This prediction does not indicate congestion, "
            "capacity problems, or a confirmed network fault."
        ),
    }


# ============================================================
# ML5 DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ML5 - MODEL PREDICTION SERVICE")
    print("=" * 70)

    print()
    print("[1] Loading ML4 model artifact...")

    artifact = load_model_artifact()

    print(
        f"Model name    : {artifact['model_name']}"
    )

    print(
        f"Model version : {artifact['model_version']}"
    )

    print(
        f"Features      : {artifact['feature_columns']}"
    )

    print(
        f"Prediction    : {artifact.get('prediction_horizon', 'N/A')}"
    )

    print()
    print("[2] Running Grid 4821 test prediction...")

    result = predict_risk(
        avg_activity=360.734596,
        activity_growth=-0.030814,
        active_hours=24,
        peak_ratio=1.848297,
        variability=147.182911,
        internet_share=0.828027,
    )

    print()
    print("Prediction result:")
    print("-" * 40)

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

    print()
    print("=" * 70)
    print("ML5 MODEL PREDICTION SERVICE: PASSED")
    print("=" * 70)