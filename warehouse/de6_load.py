import os
import shutil
import sqlite3
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT = BASE_DIR / "data" / "analytics" / "hourly_grid_summary"
STAGING = BASE_DIR / "warehouse" / "de6_staging"
DB = BASE_DIR / "warehouse" / "network_analytics.db"
SCHEMA = BASE_DIR / "warehouse" / "de6_schema.sql"

print("=" * 70)
print("DE6 - WAREHOUSE MODELLING")
print("=" * 70)

if not INPUT.exists():
    raise FileNotFoundError(f"Input does not exist: {INPUT}")

if STAGING.exists():
    shutil.rmtree(STAGING)

STAGING.mkdir(parents=True)

spark = (
    SparkSession.builder
    .appName("DE6_Warehouse_Load")
    .master("local[*]")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

print("\n[1] Loading hourly_grid_summary...")
df = spark.read.parquet(str(INPUT))

print("Columns:")
print(df.columns)

required = [
    "timestamp",
    "grid_id",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet_activity",
    "total_activity",
    "geometry",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.select(required)

print("\n[2] Preparing timestamp...")
df = df.withColumn(
    "timestamp",
    F.to_timestamp("timestamp")
)

df = df.withColumn(
    "grid_id",
    F.col("grid_id").cast("integer")
)

print("\n[3] Creating dim_time...")

timestamps = (
    df.select("timestamp")
      .distinct()
      .orderBy("timestamp")
)

window = Window.orderBy("timestamp")

dim_time = (
    timestamps
    .withColumn("time_id", F.row_number().over(window))
    .withColumn("date", F.to_date("timestamp"))
    .withColumn("hour", F.hour("timestamp"))
    .withColumn("day_of_week", F.dayofweek("timestamp"))
    .select(
        "time_id",
        "timestamp",
        "date",
        "hour",
        "day_of_week"
    )
)

time_count = dim_time.count()
print(f"dim_time rows: {time_count:,}")

dim_time.write.mode("overwrite").option("header", "true").csv(
    str(STAGING / "dim_time")
)

print("\n[4] Creating dim_grid...")

dim_grid = (
    df.select("grid_id", "geometry")
      .dropDuplicates(["grid_id"])
      .orderBy("grid_id")
)

grid_count = dim_grid.count()
print(f"dim_grid rows: {grid_count:,}")

dim_grid.write.mode("overwrite").option("header", "true").csv(
    str(STAGING / "dim_grid")
)

print("\n[5] Creating fact_network_activity...")

time_lookup = dim_time.select("timestamp", "time_id")

fact = (
    df.drop("geometry")
      .join(time_lookup, on="timestamp", how="inner")
      .select(
          "time_id",
          "grid_id",
          "sms_in",
          "sms_out",
          "call_in",
          "call_out",
          "internet_activity",
          "total_activity"
      )
)

fact_count = fact.count()
print(f"fact rows: {fact_count:,}")

fact.write.mode("overwrite").option("header", "true").csv(
    str(STAGING / "fact_network_activity")
)

spark.stop()

print("\n[6] Loading SQLite warehouse...")

if DB.exists():
    DB.unlink()

connection = sqlite3.connect(DB)
cursor = connection.cursor()

with open(SCHEMA, "r", encoding="utf-8") as f:
    cursor.executescript(f.read())

def csv_files(directory):
    return sorted(
        p for p in Path(directory).glob("*.csv")
        if p.is_file()
    )

print("\n[7] Loading dim_time...")

for file in csv_files(STAGING / "dim_time"):
    import csv

    with open(file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        batch = []

        for row in reader:
            batch.append((
                int(row["time_id"]),
                row["timestamp"],
                row["date"],
                int(row["hour"]),
                int(row["day_of_week"]),
            ))

            if len(batch) >= 5000:
                cursor.executemany(
                    """
                    INSERT INTO dim_time
                    (time_id, timestamp, date, hour, day_of_week)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                connection.commit()
                batch.clear()

        if batch:
            cursor.executemany(
                """
                INSERT INTO dim_time
                (time_id, timestamp, date, hour, day_of_week)
                VALUES (?, ?, ?, ?, ?)
                """,
                batch,
            )

connection.commit()

print("\n[8] Loading dim_grid...")

for file in csv_files(STAGING / "dim_grid"):
    import csv

    with open(file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        batch = []

        for row in reader:
            batch.append((
                int(row["grid_id"]),
                row["geometry"],
            ))

            if len(batch) >= 5000:
                cursor.executemany(
                    """
                    INSERT INTO dim_grid
                    (grid_id, geometry)
                    VALUES (?, ?)
                    """,
                    batch,
                )
                connection.commit()
                batch.clear()

        if batch:
            cursor.executemany(
                """
                INSERT INTO dim_grid
                (grid_id, geometry)
                VALUES (?, ?)
                """,
                batch,
            )

connection.commit()

print("\n[9] Loading fact_network_activity...")

import csv

total_inserted = 0

for file in csv_files(STAGING / "fact_network_activity"):
    with open(file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        batch = []

        for row in reader:
            batch.append((
                int(row["time_id"]),
                int(row["grid_id"]),
                float(row["sms_in"]),
                float(row["sms_out"]),
                float(row["call_in"]),
                float(row["call_out"]),
                float(row["internet_activity"]),
                float(row["total_activity"]),
            ))

            if len(batch) >= 10000:
                cursor.executemany(
                    """
                    INSERT INTO fact_network_activity
                    (
                        time_id,
                        grid_id,
                        sms_in,
                        sms_out,
                        call_in,
                        call_out,
                        internet_activity,
                        total_activity
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )

                connection.commit()
                total_inserted += len(batch)

                if total_inserted % 500000 == 0:
                    print(f"  inserted: {total_inserted:,}")

                batch.clear()

        if batch:
            cursor.executemany(
                """
                INSERT INTO fact_network_activity
                (
                    time_id,
                    grid_id,
                    sms_in,
                    sms_out,
                    call_in,
                    call_out,
                    internet_activity,
                    total_activity
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )

            connection.commit()
            total_inserted += len(batch)

connection.commit()

print(f"\nTotal fact rows inserted: {total_inserted:,}")

print("\n[10] Validating warehouse...")

checks = {
    "dim_time": "SELECT COUNT(*) FROM dim_time",
    "dim_grid": "SELECT COUNT(*) FROM dim_grid",
    "fact_network_activity": "SELECT COUNT(*) FROM fact_network_activity",
}

for table, query in checks.items():
    count = cursor.execute(query).fetchone()[0]
    print(f"{table}: {count:,}")

orphan_time = cursor.execute("""
SELECT COUNT(*)
FROM fact_network_activity f
LEFT JOIN dim_time t
ON f.time_id = t.time_id
WHERE t.time_id IS NULL
""").fetchone()[0]

orphan_grid = cursor.execute("""
SELECT COUNT(*)
FROM fact_network_activity f
LEFT JOIN dim_grid g
ON f.grid_id = g.grid_id
WHERE g.grid_id IS NULL
""").fetchone()[0]

print(f"\nOrphan time keys: {orphan_time}")
print(f"Orphan grid keys: {orphan_grid}")

print("\n[11] Sample fact rows...")

for row in cursor.execute("""
SELECT
    f.time_id,
    f.grid_id,
    f.sms_in,
    f.sms_out,
    f.call_in,
    f.call_out,
    f.internet_activity,
    f.total_activity
FROM fact_network_activity f
LIMIT 5
"""):
    print(row)

connection.close()

print("\n" + "=" * 70)
print("DE6 WAREHOUSE LOAD COMPLETED")
print(f"Database: {DB}")
print("=" * 70)
