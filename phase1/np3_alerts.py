import logging
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "../outputs/grid_hour_activity.csv"
)

OUTPUT_DIR = Path(
    "../outputs"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "network_alerts.csv"
)

LOG_FILE = (
    OUTPUT_DIR / "np3_alerts.log"
)


# ============================================================
# ALERT CONFIGURATION
# ============================================================

HIGH_ACTIVITY_PERCENTILE = 0.95

HIGH_INTERNET_PERCENTILE = 0.95

LOW_ACTIVITY_PERCENTILE = 0.05

SPIKE_MULTIPLIER = 2.0


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )
)

logger = logging.getLogger("NP3")


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("NP3 - RULE-BASED NETWORK ALERT GENERATOR")
print("=" * 75)


# ============================================================
# 1. LOAD GRID/HOUR DATA
# ============================================================

print("\n[1] Loading grid/hour activity data...")

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}\n"
        "Run np2_processor.py first."
    )

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

logger.info(
    "Loaded grid/hour dataset | rows=%d",
    len(df)
)


# ============================================================
# 2. VALIDATE REQUIRED COLUMNS
# ============================================================

print("\n[2] Validating required columns...")

required_columns = [
    "timestamp",
    "grid_id",
    "total_activity",
    "internet_activity"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        f"{missing_columns}"
    )

print("Status: PASS")


# ============================================================
# 3. PREPARE DATA
# ============================================================

print("\n[3] Preparing data...")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df = df.sort_values(
    ["grid_id", "timestamp"]
).reset_index(drop=True)


# ============================================================
# 4. CALCULATE THRESHOLDS
# ============================================================

print("\n[4] Calculating alert thresholds...")

high_activity_threshold = (
    df["total_activity"]
    .quantile(HIGH_ACTIVITY_PERCENTILE)
)

high_internet_threshold = (
    df["internet_activity"]
    .quantile(HIGH_INTERNET_PERCENTILE)
)

low_activity_threshold = (
    df["total_activity"]
    .quantile(LOW_ACTIVITY_PERCENTILE)
)

print(
    f"High activity threshold  "
    f"(95th percentile): "
    f"{high_activity_threshold:.4f}"
)

print(
    f"High internet threshold  "
    f"(95th percentile): "
    f"{high_internet_threshold:.4f}"
)

print(
    f"Low activity threshold   "
    f"(5th percentile): "
    f"{low_activity_threshold:.4f}"
)

print(
    f"Spike multiplier: "
    f"{SPIKE_MULTIPLIER}x"
)


# ============================================================
# 5. CALCULATE PREVIOUS-HOUR ACTIVITY
# ============================================================

print("\n[5] Calculating previous-hour activity...")

df["previous_activity"] = (
    df.groupby("grid_id")["total_activity"]
    .shift(1)
)


# ============================================================
# 6. GENERATE ALERTS
# ============================================================

print("\n[6] Generating alerts...")

alerts = []


# ------------------------------------------------------------
# Rule 1: HIGH ACTIVITY
# ------------------------------------------------------------

high_activity = (
    df["total_activity"]
    > high_activity_threshold
)

for _, row in df.loc[
    high_activity
].iterrows():

    alerts.append({
        "timestamp": row["timestamp"],
        "grid_id": int(row["grid_id"]),
        "alert_type": "HIGH_ACTIVITY",
        "severity": "HIGH",
        "metric": "total_activity",
        "value": row["total_activity"],
        "threshold": high_activity_threshold
    })


# ------------------------------------------------------------
# Rule 2: HIGH INTERNET
# ------------------------------------------------------------

high_internet = (
    df["internet_activity"]
    > high_internet_threshold
)

for _, row in df.loc[
    high_internet
].iterrows():

    alerts.append({
        "timestamp": row["timestamp"],
        "grid_id": int(row["grid_id"]),
        "alert_type": "HIGH_INTERNET",
        "severity": "MEDIUM",
        "metric": "internet_activity",
        "value": row["internet_activity"],
        "threshold": high_internet_threshold
    })


# ------------------------------------------------------------
# Rule 3: SUDDEN SPIKE
# ------------------------------------------------------------

spike_condition = (
    df["previous_activity"].notna()
    &
    (
        df["total_activity"]
        > df["previous_activity"]
        * SPIKE_MULTIPLIER
    )
)

for _, row in df.loc[
    spike_condition
].iterrows():

    alerts.append({
        "timestamp": row["timestamp"],
        "grid_id": int(row["grid_id"]),
        "alert_type": "SUDDEN_SPIKE",
        "severity": "HIGH",
        "metric": "total_activity",
        "value": row["total_activity"],
        "threshold": (
            row["previous_activity"]
            * SPIKE_MULTIPLIER
        )
    })


# ------------------------------------------------------------
# Rule 4: LOW ACTIVITY
# ------------------------------------------------------------

low_activity = (
    df["total_activity"]
    < low_activity_threshold
)

for _, row in df.loc[
    low_activity
].iterrows():

    alerts.append({
        "timestamp": row["timestamp"],
        "grid_id": int(row["grid_id"]),
        "alert_type": "LOW_ACTIVITY",
        "severity": "LOW",
        "metric": "total_activity",
        "value": row["total_activity"],
        "threshold": low_activity_threshold
    })


# ============================================================
# 7. CREATE ALERT DATAFRAME
# ============================================================

print("\n[7] Creating alert dataset...")

alerts_df = pd.DataFrame(
    alerts,
    columns=[
        "timestamp",
        "grid_id",
        "alert_type",
        "severity",
        "metric",
        "value",
        "threshold"
    ]
)


# ============================================================
# 8. ADD ALERT IDs
# ============================================================

if not alerts_df.empty:

    alerts_df.insert(
        0,
        "alert_id",
        range(
            1,
            len(alerts_df) + 1
        )
    )

    alerts_df = alerts_df.sort_values(
        ["timestamp", "grid_id", "alert_type"]
    ).reset_index(drop=True)

else:

    alerts_df.insert(
        0,
        "alert_id",
        pd.Series(dtype="int64")
    )


# ============================================================
# 9. ALERT SUMMARY
# ============================================================

print("\n[8] Alert summary")

print(
    f"Total alerts: "
    f"{len(alerts_df):,}"
)

if not alerts_df.empty:

    print("\nAlerts by type:")

    print(
        alerts_df[
            "alert_type"
        ]
        .value_counts()
        .to_string()
    )

    print("\nAlerts by severity:")

    print(
        alerts_df[
            "severity"
        ]
        .value_counts()
        .to_string()
    )

else:

    print("No alerts generated.")


# ============================================================
# 10. EXPORT
# ============================================================

print("\n[9] Exporting alerts...")

alerts_df.to_csv(
    OUTPUT_FILE,
    index=False
)

logger.info(
    "Alerts generated: %d",
    len(alerts_df)
)

logger.info(
    "Output file: %s",
    OUTPUT_FILE
)

print(
    f"Output: {OUTPUT_FILE}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("NP3 ALERT GENERATION COMPLETE")
print("=" * 75)