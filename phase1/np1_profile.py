import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

FILE_PATH = Path("C:\\Users\\jaideep.d\\network-intelligence\\data\\raw\\sms-call-internet-mi-2013-11-01.csv")


# ============================================================
# CONSTANTS
# ============================================================

ACTIVITY_COLUMNS = [
    "smsin",
    "smsout",
    "callin",
    "callout",
    "internet"
]

RAW_KEY_COLUMNS = [
    "datetime",
    "CellID",
    "countrycode"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("NP1 - MILAN TELECOM DATA PROFILING")
print("=" * 75)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading dataset...")

df = pd.read_csv(FILE_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 2. COLUMN INFORMATION
# ============================================================

print("\n[2] Column names")

for column in df.columns:
    print(f"  - {column}")


# ============================================================
# 3. DATA TYPES
# ============================================================

print("\n[3] Data types")

print(df.dtypes)


# ============================================================
# 4. FIRST RECORDS
# ============================================================

print("\n[4] First 5 records")

print(df.head().to_string())


# ============================================================
# 5. TIMESTAMP CONVERSION
# ============================================================

print("\n[5] Timestamp analysis")

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)

print(f"Minimum timestamp : {df['datetime'].min()}")
print(f"Maximum timestamp : {df['datetime'].max()}")
print(f"Unique timestamps : {df['datetime'].nunique()}")

print(
    f"Unique dates      : "
    f"{df['datetime'].dt.date.nunique()}"
)


# ============================================================
# 6. CHECK HOURLY INTERVALS
# ============================================================

print("\n[6] Hourly interval validation")

hours = sorted(
    df["datetime"]
    .dt.hour
    .dropna()
    .unique()
)

print("Hours found:", hours)
print("Number of unique hours:", len(hours))

if len(hours) == 24:
    print("Status: PASS - 24 hourly intervals found")
else:
    print("Status: CHECK - Expected 24 hourly intervals")


# ============================================================
# 7. NULL VALUE ANALYSIS
# ============================================================

print("\n[7] Null value analysis")

null_counts = df.isnull().sum()

null_percentage = (
    df.isnull().mean() * 100
).round(2)

null_report = pd.DataFrame({
    "null_count": null_counts,
    "null_percentage": null_percentage
})

print(null_report)


# ============================================================
# 8. ACTIVITY NULL ANALYSIS
# ============================================================

print("\n[8] Activity field analysis")

for column in ACTIVITY_COLUMNS:

    null_count = df[column].isna().sum()
    non_null_count = df[column].notna().sum()

    print(
        f"{column:10} | "
        f"non-null: {non_null_count:>9,} | "
        f"null: {null_count:>9,} | "
        f"null %: {null_count / len(df) * 100:6.2f}%"
    )


# ============================================================
# 9. COMPLETE ROW DUPLICATES
# ============================================================

print("\n[9] Complete duplicate records")

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count:,}")

if duplicate_count == 0:
    print("Status: PASS")
else:
    print("Status: CHECK")


# ============================================================
# 10. RAW GRAIN DUPLICATES
# ============================================================

print("\n[10] Raw grain validation")

print("Expected raw grain:")
print("datetime + CellID + countrycode")

raw_key_duplicates = df.duplicated(
    subset=RAW_KEY_COLUMNS
).sum()

print(
    f"Duplicate raw keys: "
    f"{raw_key_duplicates:,}"
)

if raw_key_duplicates == 0:
    print("Status: PASS - raw grain is unique")
else:
    print("Status: FAIL - duplicate raw keys found")


# ============================================================
# 11. UNIQUE NETWORK DIMENSIONS
# ============================================================

print("\n[11] Network dimensions")

print(
    f"Unique CellIDs     : "
    f"{df['CellID'].nunique():,}"
)

print(
    f"Unique countrycodes: "
    f"{df['countrycode'].nunique():,}"
)


# ============================================================
# 12. NEGATIVE ACTIVITY CHECK
# ============================================================

print("\n[12] Negative activity validation")

negative_found = False

for column in ACTIVITY_COLUMNS:

    negative_count = (
        df[column].dropna() < 0
    ).sum()

    print(
        f"{column:10}: "
        f"{negative_count:,} negative values"
    )

    if negative_count > 0:
        negative_found = True

if negative_found:
    print("Status: FAIL - negative activity detected")
else:
    print("Status: PASS - no negative activity")


# ============================================================
# 13. ACTIVITY STATISTICS
# ============================================================

print("\n[13] Activity statistics")

activity_stats = df[ACTIVITY_COLUMNS].describe().T

print(
    activity_stats[
        ["count", "mean", "std", "min", "50%", "max"]
    ].to_string()
)


# ============================================================
# 14. DERIVED ACTIVITY METRICS
# ============================================================

print("\n[14] Derived activity metrics")

# For profiling purposes, treat missing activity values
# as zero when calculating aggregate activity measures.

df["total_sms"] = (
    df["smsin"].fillna(0)
    + df["smsout"].fillna(0)
)

df["total_calls"] = (
    df["callin"].fillna(0)
    + df["callout"].fillna(0)
)

df["total_activity"] = (
    df["total_sms"]
    + df["total_calls"]
    + df["internet"].fillna(0)
)

print(
    f"Total SMS activity    : "
    f"{df['total_sms'].sum():,.4f}"
)

print(
    f"Total call activity   : "
    f"{df['total_calls'].sum():,.4f}"
)

print(
    f"Total network activity: "
    f"{df['total_activity'].sum():,.4f}"
)


# ============================================================
# 15. ACTIVITY SUMMARY
# ============================================================

print("\n[15] Activity summary")

for column in ACTIVITY_COLUMNS + [
    "total_sms",
    "total_calls",
    "total_activity"
]:

    print(
        f"{column:15} | "
        f"min={df[column].min():,.4f} | "
        f"max={df[column].max():,.4f} | "
        f"mean={df[column].mean():,.4f}"
    )


# ============================================================
# 16. PROFILE SUMMARY
# ============================================================

print("\n[16] Profiling summary")

print(f"Rows                  : {len(df):,}")
print(f"Columns               : {len(df.columns) - 3:,}")
print(f"Unique CellIDs        : {df['CellID'].nunique():,}")
print(f"Unique countrycodes   : {df['countrycode'].nunique():,}")
print(f"Unique timestamps     : {df['datetime'].nunique():,}")
print(f"Complete duplicates   : {duplicate_count:,}")
print(f"Raw grain duplicates  : {raw_key_duplicates:,}")

print(
    f"Negative activity     : "
    f"{'YES' if negative_found else 'NO'}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("NP1 PROFILING COMPLETE")
print("=" * 75)