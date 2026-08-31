import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    countDistinct,
    hour,
    to_timestamp,
    input_file_name,
    regexp_extract,
    min as spark_min,
    max as spark_max
)


HADOOP_HOME = os.path.join(
    os.path.expanduser("~"),
    "hadoop"
)

os.environ["HADOOP_HOME"] = HADOOP_HOME
os.environ["hadoop.home.dir"] = HADOOP_HOME

hadoop_bin = os.path.join(
    HADOOP_HOME,
    "bin"
)

os.environ["PATH"] = (
    hadoop_bin
    + os.pathsep
    + os.environ.get("PATH", "")
)

os.environ["HADOOP_HOME"] = HADOOP_HOME
os.environ["hadoop.home.dir"] = HADOOP_HOME
# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INPUT_PATTERN = str(
    RAW_DIR / "sms-call-internet-mi-*.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "sp1_ingestion_summary.txt"
)


# ============================================================
# SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("MilanTelecomIngestion")
    .config(
        "spark.driver.extraJavaOptions",
        f"-Djava.library.path={hadoop_bin}"
    )
    .config(
        "spark.executor.extraJavaOptions",
        f"-Djava.library.path={hadoop_bin}"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("SP1 - DISTRIBUTED MILAN TELECOM DATA INGESTION")
print("=" * 75)


# ============================================================
# 1. DISCOVER INPUT FILES
# ============================================================

print("\n[1] Discovering Milan input files...")

input_files = sorted(
    RAW_DIR.glob(
        "sms-call-internet-mi-*.csv"
    )
)

if not input_files:
    raise FileNotFoundError(
        f"No Milan CSV files found in: {RAW_DIR}"
    )

print(
    f"Files found: {len(input_files)}"
)

for file in input_files:
    print(
        f"  - {file.name}"
    )


# ============================================================
# 2. READ ALL FILES AS ONE SPARK DATAFRAME
# ============================================================

print("\n[2] Reading all files with Spark...")
input_files = sorted(
    str(path.resolve())
    for path in RAW_DIR.glob("sms-call-internet-mi-*.csv")
)

if not input_files:
    raise FileNotFoundError(
        f"No Milan telecom CSV files found in: {RAW_DIR}"
    )

print(f"Files found: {len(input_files)}")

for file in input_files:
    print(f"  - {Path(file).name}")

print("\n[2] Reading all files with Spark...")

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(input_files)
)

print("Status: PASS - files loaded")


# ============================================================
# 3. ADD SOURCE FILE INFORMATION
# ============================================================

print("\n[3] Adding source-file metadata...")

df = df.withColumn(
    "source_file",
    input_file_name()
)

df = df.withColumn(
    "source_filename",
    regexp_extract(
        col("source_file"),
        r"([^/\\]+)$",
        1
    )
)


# ============================================================
# 4. DISPLAY SCHEMA
# ============================================================

print("\n[4] Spark DataFrame schema")

df.printSchema()


# ============================================================
# 5. BASIC DATASET STATISTICS
# ============================================================

print("\n[5] Dataset statistics...")

row_count = df.count()

column_count = len(
    df.columns
)

partition_count = (
    df.rdd.getNumPartitions()
)

print(
    f"Rows        : {row_count:,}"
)

print(
    f"Columns     : {column_count}"
)

print(
    f"Partitions  : {partition_count}"
)


# ============================================================
# 6. EXPECTED SCHEMA VALIDATION
# ============================================================

print("\n[6] Validating expected columns...")

expected_columns = [
    "datetime",
    "CellID",
    "countrycode",
    "smsin",
    "smsout",
    "callin",
    "callout",
    "internet"
]

missing_columns = [
    c
    for c in expected_columns
    if c not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

print(
    "Status: PASS - expected columns found"
)


# ============================================================
# 7. SOURCE FILE VALIDATION
# ============================================================

print("\n[7] Validating source files...")

source_summary = (
    df
    .groupBy("source_filename")
    .agg(
        count("*").alias("row_count")
    )
    .orderBy("source_filename")
)

source_summary.show(
    20,
    truncate=False
)

source_file_count = (
    df
    .select(
        "source_filename"
    )
    .distinct()
    .count()
)

print(
    f"Distinct source files: "
    f"{source_file_count}"
)

if source_file_count != len(input_files):

    raise ValueError(
        "Source file count mismatch"
    )

print(
    "Status: PASS"
)


# ============================================================
# 8. TIMESTAMP CONVERSION
# ============================================================

print("\n[8] Validating timestamps...")

df = df.withColumn(
    "timestamp",
    to_timestamp(
        col("datetime"),
        "yyyy-MM-dd HH:mm:ss"
    )
)

invalid_timestamp_count = (
    df
    .filter(
        col("timestamp").isNull()
        &
        col("datetime").isNotNull()
    )
    .count()
)

print(
    f"Invalid timestamps: "
    f"{invalid_timestamp_count:,}"
)

if invalid_timestamp_count > 0:

    print(
        "Status: FAIL - invalid timestamps found"
    )

    raise ValueError(
        "Invalid timestamp values detected"
    )

print(
    "Status: PASS"
)


# ============================================================
# 9. DATE COVERAGE
# ============================================================

print("\n[9] Checking date coverage...")

date_summary = (
    df
    .select(
        col("timestamp").cast("date").alias("date")
    )
    .groupBy("date")
    .agg(
        count("*").alias("row_count")
    )
    .orderBy("date")
)

date_summary.show(
    20,
    truncate=False
)

date_count = (
    df
    .select(
        col("timestamp").cast("date")
        .alias("date")
    )
    .distinct()
    .count()
)

print(
    f"Unique dates: {date_count}"
)


# ============================================================
# 10. HOURLY COVERAGE
# ============================================================

print("\n[10] Checking hourly coverage...")

hour_summary = (
    df
    .select(
        hour("timestamp").alias("hour")
    )
    .distinct()
    .orderBy("hour")
)

hours = [
    row["hour"]
    for row in hour_summary.collect()
]

print(
    f"Hours found: {hours}"
)

print(
    f"Number of unique hours: "
    f"{len(hours)}"
)

expected_hours = list(
    range(24)
)

if hours == expected_hours:

    print(
        "Status: PASS - all 24 hourly intervals found"
    )

else:

    print(
        "Status: WARNING - hourly coverage differs"
    )


# ============================================================
# 11. GRID COVERAGE
# ============================================================

print("\n[11] Checking grid coverage...")

grid_count = (
    df
    .select("CellID")
    .distinct()
    .count()
)

print(
    f"Unique CellIDs: {grid_count:,}"
)


# ============================================================
# 12. COUNTRY CODE COVERAGE
# ============================================================

print("\n[12] Checking country-code coverage...")

country_count = (
    df
    .select("countrycode")
    .distinct()
    .count()
)

print(
    f"Unique country codes: "
    f"{country_count:,}"
)


# ============================================================
# 13. RAW NULL ANALYSIS
# ============================================================

print("\n[13] Raw null analysis...")

activity_columns = [
    "smsin",
    "smsout",
    "callin",
    "callout",
    "internet"
]

for column in activity_columns:

    null_count = (
        df
        .filter(
            col(column).isNull()
        )
        .count()
    )

    print(
        f"{column:10s}: "
        f"{null_count:,}"
    )


# ============================================================
# 14. SAMPLE RECORDS
# ============================================================

print("\n[14] Sample records...")

df.select(
    "datetime",
    "CellID",
    "countrycode",
    "smsin",
    "smsout",
    "callin",
    "callout",
    "internet",
    "source_filename"
).show(
    5,
    truncate=False
)


# ============================================================
# 15. FILE-LEVEL SUMMARY
# ============================================================

print("\n[15] File-level ingestion summary")

file_summary = (
    df
    .groupBy("source_filename")
    .agg(
        count("*").alias("rows"),
        countDistinct("CellID").alias("unique_grids"),
        spark_min("timestamp").alias("min_timestamp"),
        spark_max("timestamp").alias("max_timestamp")
    )
    .orderBy("source_filename")
)

file_summary.show(
    20,
    truncate=False
)


# ============================================================
# 16. WRITE SUMMARY REPORT
# ============================================================

print("\n[16] Writing ingestion summary...")

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "SP1 - DISTRIBUTED MILAN TELECOM DATA INGESTION\n"
    )

    report.write("=" * 60 + "\n\n")

    report.write(
        f"Input directory: {RAW_DIR}\n"
    )

    report.write(
        f"Input pattern: sms-call-internet-mi-*.csv\n"
    )

    report.write(
        f"Files discovered: {len(input_files)}\n"
    )

    report.write(
        f"Rows ingested: {row_count:,}\n"
    )

    report.write(
        f"Columns: {column_count}\n"
    )

    report.write(
        f"Spark partitions: {partition_count}\n"
    )

    report.write(
        f"Unique dates: {date_count}\n"
    )

    report.write(
        f"Unique grids: {grid_count:,}\n"
    )

    report.write(
        f"Unique country codes: {country_count:,}\n"
    )

    report.write(
        f"Invalid timestamps: "
        f"{invalid_timestamp_count:,}\n"
    )

    report.write(
        f"Unique source files: "
        f"{source_file_count}\n"
    )

    report.write("\nFile-level summary:\n")

    for row in file_summary.collect():

        report.write(
            f"{row['source_filename']} | "
            f"rows={row['rows']:,} | "
            f"grids={row['unique_grids']:,} | "
            f"min={row['min_timestamp']} | "
            f"max={row['max_timestamp']}\n"
        )


# ============================================================
# 17. SPARK EXECUTION PLAN
# ============================================================

print("\n[17] Spark execution plan")

df.select(
    "CellID",
    "timestamp",
    "internet"
).limit(10).explain(
    mode="formatted"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("SP1 INGESTION COMPLETE")
print("=" * 75)

print(
    f"Files processed : {len(input_files)}"
)

print(
    f"Rows ingested   : {row_count:,}"
)

print(
    f"Partitions      : {partition_count}"
)

print(
    f"Unique dates    : {date_count}"
)

print(
    f"Unique grids    : {grid_count:,}"
)

print(
    f"Source files    : {source_file_count}"
)

print(
    f"Report          : {OUTPUT_FILE}"
)

print("=" * 75)


# ============================================================
# STOP SPARK
# ============================================================

spark.stop()