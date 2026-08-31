import os
import sys
import time

# ============================================================
# WINDOWS PYTHON WORKER CONFIGURATION
# ============================================================

PYTHON_EXE = sys.executable

os.environ["PYSPARK_PYTHON"] = PYTHON_EXE
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_EXE


from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

SP4_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "sp4_geo_enriched"
)

SP5_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "sp5_performance"
)


# ============================================================
# EXPECTATIONS
# ============================================================

EXPECTED_SP4_ROWS = 1_679_994

EXPECTED_GRID_COUNT = 10_000

EXPECTED_DAYS = 7

EXPECTED_HOURS = 24


# ============================================================
# REQUIRED SP4 COLUMNS
# ============================================================

REQUIRED_SP4_COLUMNS = [

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


# ============================================================
# HEADER
# ============================================================

def print_header(title):

    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# CREATE SPARK SESSION
# ============================================================

def create_spark_session():

    print()
    print("[1] Creating Spark session...")
    print("-" * 75)

    spark = (
        SparkSession.builder

        .appName(
            "SP5-Milan-Telecom-Performance"
        )

        .master("local[*]")

        # ----------------------------------------------------
        # Shuffle configuration
        # ----------------------------------------------------

        .config(
            "spark.sql.shuffle.partitions",
            "8"
        )

        # ----------------------------------------------------
        # Adaptive Query Execution
        # ----------------------------------------------------

        .config(
            "spark.sql.adaptive.enabled",
            "true"
        )

        .config(
            "spark.sql.adaptive.coalescePartitions.enabled",
            "true"
        )

        .config(
            "spark.sql.adaptive.skewJoin.enabled",
            "true"
        )

        # ----------------------------------------------------
        # Broadcast threshold
        # ----------------------------------------------------

        .config(
            "spark.sql.autoBroadcastJoinThreshold",
            "50MB"
        )

        # ----------------------------------------------------
        # Python worker configuration
        # ----------------------------------------------------

        .config(
            "spark.pyspark.python",
            PYTHON_EXE
        )

        .config(
            "spark.pyspark.driver.python",
            PYTHON_EXE
        )

        .config(
            "spark.python.worker.reuse",
            "true"
        )

        # ----------------------------------------------------
        # Time zone
        # ----------------------------------------------------

        .config(
            "spark.sql.session.timeZone",
            "UTC"
        )

        # ----------------------------------------------------
        # Debug
        # ----------------------------------------------------

        .config(
            "spark.sql.debug.maxToStringFields",
            "200"
        )

        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(
        "WARN"
    )

    print(
        "Spark session created successfully"
    )

    print(
        f"Spark version: {spark.version}"
    )

    print(
        f"Python executable: {PYTHON_EXE}"
    )

    return spark


# ============================================================
# CHECK SP4 OUTPUT
# ============================================================

def validate_sp4_output_exists():

    print()
    print("[2] Checking SP4 Parquet input...")
    print("-" * 75)

    if not os.path.exists(
        SP4_OUTPUT_DIR
    ):

        raise FileNotFoundError(
            "SP4 output directory does not exist:\n"
            f"{SP4_OUTPUT_DIR}\n\n"
            "Run SP4 first."
        )

    parquet_files = []

    for root, dirs, files in os.walk(
        SP4_OUTPUT_DIR
    ):

        for filename in files:

            if filename.lower().endswith(
                ".parquet"
            ):

                parquet_files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    print(
        f"SP4 input directory: "
        f"{SP4_OUTPUT_DIR}"
    )

    print(
        f"Parquet files found: "
        f"{len(parquet_files)}"
    )

    if not parquet_files:

        raise FileNotFoundError(
            "No Parquet files found in SP4 output."
        )

    print(
        "SP4 Parquet input: PASS"
    )


# ============================================================
# LOAD SP4
# ============================================================

def load_sp4(spark):

    print()
    print("[3] Reading SP4 GeoJSON-enriched Parquet...")
    print("-" * 75)

    start_time = time.perf_counter()

    df = (
        spark.read
        .parquet(
            SP4_OUTPUT_DIR
        )
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Read operation completed in "
        f"{elapsed:.3f} seconds"
    )

    print()
    print(
        f"SP4 columns: "
        f"{len(df.columns)}"
    )

    print()
    print(
        "SP4 schema:"
    )

    df.printSchema()

    return df


# ============================================================
# VALIDATE SP4 SCHEMA
# ============================================================

def validate_sp4_schema(df):

    print()
    print("[4] Validating SP4 schema...")
    print("-" * 75)

    missing = [

        column

        for column in REQUIRED_SP4_COLUMNS

        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "SP4 is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    print(
        "All required SP4 columns found."
    )

    print(
        "Schema validation: PASS"
    )


# ============================================================
# ROW COUNT VALIDATION
# ============================================================

def validate_sp4_row_count(df):

    print()
    print("[5] Validating SP4 row count...")
    print("-" * 75)

    row_count = df.count()

    print(
        f"SP4 rows: {row_count:,}"
    )

    print(
        f"Expected: {EXPECTED_SP4_ROWS:,}"
    )

    if row_count != EXPECTED_SP4_ROWS:

        raise ValueError(
            "SP4 row count mismatch. "
            f"Expected {EXPECTED_SP4_ROWS:,}, "
            f"found {row_count:,}."
        )

    print(
        "SP4 row-count validation: PASS"
    )

    return row_count


# ============================================================
# INSPECT PARTITIONS
# ============================================================

def inspect_partitions(df):

    print()
    print("[6] Inspecting Spark partitions...")
    print("-" * 75)

    partition_count = (
        df.rdd.getNumPartitions()
    )

    print(
        f"Current DataFrame partitions: "
        f"{partition_count}"
    )

    if partition_count <= 0:

        raise ValueError(
            "Invalid partition count."
        )

    print(
        "Partition inspection: PASS"
    )

    return partition_count


# ============================================================
# PARTITION DISTRIBUTION
# ============================================================

def inspect_partition_distribution(df):

    print()
    print("[7] Inspecting partition distribution...")
    print("-" * 75)

    partition_distribution = (
        df

        .withColumn(
            "_partition_id",
            F.spark_partition_id()
        )

        .groupBy(
            "_partition_id"
        )

        .count()

        .orderBy(
            "_partition_id"
        )
    )

    print()
    print(
        "Rows per Spark partition:"
    )

    partition_distribution.show(
        100,
        truncate=False
    )

    print(
        "Partition distribution inspection: PASS"
    )


# ============================================================
# CACHE PERFORMANCE DATAFRAME
# ============================================================

def cache_dataframe(df):

    print()
    print("[8] Caching SP4 DataFrame...")
    print("-" * 75)

    start_time = time.perf_counter()

    df.cache()

    cached_count = df.count()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Cached rows: {cached_count:,}"
    )

    print(
        f"Cache materialization time: "
        f"{elapsed:.3f} seconds"
    )

    print(
        "SP4 DataFrame cache: PASS"
    )

    return df


# ============================================================
# EXECUTION PLAN
# ============================================================

def inspect_execution_plan(df):

    print()
    print("[9] Inspecting Spark execution plan...")
    print("-" * 75)

    print()
    print(
        "FORMATTED EXECUTION PLAN"
    )

    print(
        "-" * 75
    )

    (
        df

        .select(
            "timestamp",
            "grid_id",
            "total_activity"
        )

        .groupBy(
            "grid_id"
        )

        .agg(
            F.sum(
                "total_activity"
            ).alias(
                "grid_total_activity"
            )
        )

        .orderBy(
            F.desc(
                "grid_total_activity"
            )
        )

        .explain(
            "formatted"
        )
    )

    print()
    print(
        "Execution-plan inspection: PASS"
    )


# ============================================================
# GROUPBY PERFORMANCE TEST
# ============================================================

def benchmark_grid_aggregation(df):

    print()
    print("[10] Benchmarking grid-level aggregation...")
    print("-" * 75)

    print()
    print(
        "Operation:"
    )

    print(
        "GROUP BY grid_id"
    )

    print(
        "SUM(total_activity)"
    )

    start_time = time.perf_counter()

    result = (

        df

        .groupBy(
            "grid_id"
        )

        .agg(

            F.sum(
                "sms_in"
            ).alias(
                "window_sms_in"
            ),

            F.sum(
                "sms_out"
            ).alias(
                "window_sms_out"
            ),

            F.sum(
                "call_in"
            ).alias(
                "window_call_in"
            ),

            F.sum(
                "call_out"
            ).alias(
                "window_call_out"
            ),

            F.sum(
                "internet_activity"
            ).alias(
                "window_internet_activity"
            ),

            F.sum(
                "total_activity"
            ).alias(
                "window_total_activity"
            )
        )
    )

    result_count = result.count()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print(
        f"Aggregation result rows: "
        f"{result_count:,}"
    )

    print(
        f"Aggregation execution time: "
        f"{elapsed:.3f} seconds"
    )

    if result_count != EXPECTED_GRID_COUNT:

        raise ValueError(
            "Expected "
            f"{EXPECTED_GRID_COUNT:,} "
            "grid-level aggregation rows, "
            f"found {result_count:,}."
        )

    print(
        "Grid aggregation: PASS"
    )

    return result, elapsed


# ============================================================
# TEMPORAL AGGREGATION PERFORMANCE
# ============================================================

def benchmark_temporal_aggregation(df):

    print()
    print("[11] Benchmarking temporal aggregation...")
    print("-" * 75)

    start_time = time.perf_counter()

    result = (

        df

        .groupBy(
            "timestamp"
        )

        .agg(

            F.sum(
                "total_activity"
            ).alias(
                "timestamp_total_activity"
            )
        )

        .orderBy(
            "timestamp"
        )
    )

    result_count = result.count()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Distinct timestamps: "
        f"{result_count:,}"
    )

    print(
        f"Temporal aggregation time: "
        f"{elapsed:.3f} seconds"
    )

    expected_timestamps = (
        EXPECTED_DAYS
        *
        EXPECTED_HOURS
    )

    print(
        f"Expected timestamps: "
        f"{expected_timestamps:,}"
    )

    if result_count != expected_timestamps:

        raise ValueError(
            "Temporal coverage mismatch. "
            f"Expected {expected_timestamps:,}, "
            f"found {result_count:,}."
        )

    print(
        "Temporal aggregation: PASS"
    )

    return result, elapsed


# ============================================================
# TOP ACTIVITY PERFORMANCE
# ============================================================

def benchmark_top_activity(df):

    print()
    print("[12] Benchmarking top-activity query...")
    print("-" * 75)

    start_time = time.perf_counter()

    top_activity = (

        df

        .groupBy(
            "grid_id"
        )

        .agg(

            F.sum(
                "total_activity"
            ).alias(
                "window_total_activity"
            )
        )

        .orderBy(
            F.desc(
                "window_total_activity"
            )
        )

        .limit(10)
    )

    top_rows = top_activity.collect()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print(
        "Top 10 grids:"
    )

    for row in top_rows:

        print(
            f"grid_id={row['grid_id']} | "
            f"activity="
            f"{row['window_total_activity']}"
        )

    print()
    print(
        f"Top-activity query time: "
        f"{elapsed:.3f} seconds"
    )

    if len(top_rows) != 10:

        raise ValueError(
            "Top-activity query did not "
            "return 10 rows."
        )

    print(
        "Top-activity benchmark: PASS"
    )

    return elapsed


# ============================================================
# SHUFFLE / REPARTITION TEST
# ============================================================

def benchmark_repartition(df):

    print()
    print("[13] Benchmarking repartition operation...")
    print("-" * 75)

    current_partitions = (
        df.rdd.getNumPartitions()
    )

    target_partitions = max(
        8,
        current_partitions
    )

    print(
        f"Current partitions: "
        f"{current_partitions}"
    )

    print(
        f"Target partitions: "
        f"{target_partitions}"
    )

    start_time = time.perf_counter()

    repartitioned = (
        df

        .repartition(
            target_partitions,
            "grid_id"
        )
    )

    repartitioned_count = (
        repartitioned.count()
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    new_partition_count = (
        repartitioned.rdd.getNumPartitions()
    )

    print(
        f"Rows after repartition: "
        f"{repartitioned_count:,}"
    )

    print(
        f"Partitions after repartition: "
        f"{new_partition_count}"
    )

    print(
        f"Repartition time: "
        f"{elapsed:.3f} seconds"
    )

    if repartitioned_count != EXPECTED_SP4_ROWS:

        raise ValueError(
            "Repartition changed row count."
        )

    print(
        "Repartition validation: PASS"
    )

    return repartitioned


# ============================================================
# COALESCE TEST
# ============================================================

def benchmark_coalesce(df):

    print()
    print("[14] Benchmarking coalesce operation...")
    print("-" * 75)

    current_partitions = (
        df.rdd.getNumPartitions()
    )

    target_partitions = max(
        1,
        current_partitions // 2
    )

    print(
        f"Current partitions: "
        f"{current_partitions}"
    )

    print(
        f"Coalesce target: "
        f"{target_partitions}"
    )

    start_time = time.perf_counter()

    coalesced = (
        df.coalesce(
            target_partitions
        )
    )

    coalesced_count = (
        coalesced.count()
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    final_partitions = (
        coalesced.rdd.getNumPartitions()
    )

    print(
        f"Rows after coalesce: "
        f"{coalesced_count:,}"
    )

    print(
        f"Partitions after coalesce: "
        f"{final_partitions}"
    )

    print(
        f"Coalesce time: "
        f"{elapsed:.3f} seconds"
    )

    if coalesced_count != EXPECTED_SP4_ROWS:

        raise ValueError(
            "Coalesce changed row count."
        )

    print(
        "Coalesce validation: PASS"
    )

    return coalesced


# ============================================================
# DATA QUALITY PERFORMANCE CHECK
# ============================================================

def performance_data_quality_check(df):

    print()
    print("[15] Running performance-stage data-quality checks...")
    print("-" * 75)

    checks = {

        "NULL timestamp":
            df.filter(
                F.col("timestamp").isNull()
            ).count(),

        "NULL grid_id":
            df.filter(
                F.col("grid_id").isNull()
            ).count(),

        "NULL geometry":
            df.filter(
                F.col("geometry").isNull()
            ).count(),

        "Negative total_activity":
            df.filter(
                F.col("total_activity") < 0
            ).count(),
    }

    failure = False

    for name, count in checks.items():

        print(
            f"{name:<30}: "
            f"{count:,}"
        )

        if count != 0:
            failure = True

    if failure:

        raise ValueError(
            "SP5 performance-stage "
            "data-quality validation failed."
        )

    print()
    print(
        "Performance-stage data quality: PASS"
    )


# ============================================================
# PARTITION BY GRID VALIDATION
# ============================================================

def validate_grid_partitioning(df):

    print()
    print("[16] Validating grid-based partitioning...")
    print("-" * 75)

    start_time = time.perf_counter()

    partitioned = (
        df

        .repartition(
            8,
            "grid_id"
        )
    )

    count = partitioned.count()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Rows after grid repartition: "
        f"{count:,}"
    )

    print(
        f"Execution time: "
        f"{elapsed:.3f} seconds"
    )

    if count != EXPECTED_SP4_ROWS:

        raise ValueError(
            "Grid partitioning changed row count."
        )

    print(
        "Grid-based repartition validation: PASS"
    )

    return partitioned


# ============================================================
# FINAL PERFORMANCE SUMMARY
# ============================================================

def show_performance_summary(
    original_partitions,
    aggregation_time,
    temporal_time,
    top_activity_time
):

    print()
    print("=" * 75)
    print("SP5 PERFORMANCE SUMMARY")
    print("=" * 75)

    print(
        f"SP4 input rows              : "
        f"{EXPECTED_SP4_ROWS:,}"
    )

    print(
        f"SP4 original partitions     : "
        f"{original_partitions}"
    )

    print(
        f"Grid aggregation time       : "
        f"{aggregation_time:.3f} seconds"
    )

    print(
        f"Temporal aggregation time   : "
        f"{temporal_time:.3f} seconds"
    )

    print(
        f"Top activity query time     : "
        f"{top_activity_time:.3f} seconds"
    )

    print(
        "Adaptive Query Execution    : ENABLED"
    )

    print(
        "Spark shuffle partitions    : 8"
    )

    print(
        "Performance validation      : PASS"
    )

    print(
        "Status                      : SUCCESS"
    )

    print("=" * 75)


# ============================================================
# WRITE SP5 PERFORMANCE OUTPUT
# ============================================================

def write_sp5_output(
    spark,
    grid_summary,
    temporal_summary
):

    print()
    print("[17] Writing SP5 performance results...")
    print("-" * 75)

    if os.path.exists(
        SP5_OUTPUT_DIR
    ):

        print(
            "Previous SP5 output exists."
        )

        print(
            "Spark will overwrite it."
        )

    # --------------------------------------------------------
    # GRID SUMMARY
    # --------------------------------------------------------

    grid_output = os.path.join(
        SP5_OUTPUT_DIR,
        "grid_performance"
    )

    (
        grid_summary

        .write

        .mode("overwrite")

        .option(
            "compression",
            "snappy"
        )

        .parquet(
            grid_output
        )
    )

    # --------------------------------------------------------
    # TEMPORAL SUMMARY
    # --------------------------------------------------------

    temporal_output = os.path.join(
        SP5_OUTPUT_DIR,
        "temporal_performance"
    )

    (
        temporal_summary

        .write

        .mode("overwrite")

        .option(
            "compression",
            "snappy"
        )

        .parquet(
            temporal_output
        )
    )

    print()
    print(
        "SP5 performance results written successfully."
    )

    print(
        f"Output directory: "
        f"{SP5_OUTPUT_DIR}"
    )

    return SP5_OUTPUT_DIR


# ============================================================
# VERIFY SP5 OUTPUT
# ============================================================

def verify_sp5_output(spark):

    print()
    print("[18] Verifying SP5 Parquet output...")
    print("-" * 75)

    grid_output = os.path.join(
        SP5_OUTPUT_DIR,
        "grid_performance"
    )

    temporal_output = os.path.join(
        SP5_OUTPUT_DIR,
        "temporal_performance"
    )

    if not os.path.exists(
        grid_output
    ):

        raise FileNotFoundError(
            "SP5 grid performance output "
            "was not created."
        )

    if not os.path.exists(
        temporal_output
    ):

        raise FileNotFoundError(
            "SP5 temporal performance output "
            "was not created."
        )

    grid_df = (
        spark.read
        .parquet(
            grid_output
        )
    )

    temporal_df = (
        spark.read
        .parquet(
            temporal_output
        )
    )

    grid_count = (
        grid_df.count()
    )

    temporal_count = (
        temporal_df.count()
    )

    print(
        f"Grid performance rows     : "
        f"{grid_count:,}"
    )

    print(
        f"Temporal performance rows : "
        f"{temporal_count:,}"
    )

    if grid_count != EXPECTED_GRID_COUNT:

        raise ValueError(
            "SP5 grid performance output "
            "row count mismatch."
        )

    expected_timestamps = (
        EXPECTED_DAYS
        *
        EXPECTED_HOURS
    )

    if temporal_count != expected_timestamps:

        raise ValueError(
            "SP5 temporal performance output "
            "row count mismatch."
        )

    print()
    print(
        "SP5 verified grid schema:"
    )

    grid_df.printSchema()

    print()
    print(
        "SP5 verified temporal schema:"
    )

    temporal_df.printSchema()

    print()
    print(
        "SP5 Parquet verification: PASS"
    )

    return (
        grid_df,
        temporal_df
    )


# ============================================================
# MAIN SP5
# ============================================================

def run_sp5():

    spark = None

    try:

        print_header(
            "SP5 - MILAN TELECOM PERFORMANCE VALIDATION"
        )

        # ====================================================
        # SPARK
        # ====================================================

        spark = (
            create_spark_session()
        )

        # ====================================================
        # SP4 INPUT
        # ====================================================

        validate_sp4_output_exists()

        activity_df = (
            load_sp4(
                spark
            )
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        validate_sp4_schema(
            activity_df
        )

        sp4_rows = (
            validate_sp4_row_count(
                activity_df
            )
        )

        # ====================================================
        # PARTITIONS
        # ====================================================

        original_partitions = (
            inspect_partitions(
                activity_df
            )
        )

        inspect_partition_distribution(
            activity_df
        )

        # ====================================================
        # CACHE
        # ====================================================

        activity_df = (
            cache_dataframe(
                activity_df
            )
        )

        # ====================================================
        # EXECUTION PLAN
        # ====================================================

        inspect_execution_plan(
            activity_df
        )

        # ====================================================
        # GRID AGGREGATION
        # ====================================================

        (
            grid_summary,
            aggregation_time
        ) = benchmark_grid_aggregation(
            activity_df
        )

        # ====================================================
        # TEMPORAL AGGREGATION
        # ====================================================

        (
            temporal_summary,
            temporal_time
        ) = benchmark_temporal_aggregation(
            activity_df
        )

        # ====================================================
        # TOP ACTIVITY
        # ====================================================

        top_activity_time = (
            benchmark_top_activity(
                activity_df
            )
        )

        # ====================================================
        # REPARTITION
        # ====================================================

        repartitioned_df = (
            benchmark_repartition(
                activity_df
            )
        )

        # ====================================================
        # COALESCE
        # ====================================================

        coalesced_df = (
            benchmark_coalesce(
                activity_df
            )
        )

        # ====================================================
        # DATA QUALITY
        # ====================================================

        performance_data_quality_check(
            activity_df
        )

        # ====================================================
        # GRID PARTITIONING
        # ====================================================

        grid_partitioned_df = (
            validate_grid_partitioning(
                activity_df
            )
        )

        # ====================================================
        # WRITE RESULTS
        # ====================================================

        write_sp5_output(
            spark,
            grid_summary,
            temporal_summary
        )

        # ====================================================
        # VERIFY RESULTS
        # ====================================================

        (
            verified_grid_df,
            verified_temporal_df
        ) = verify_sp5_output(
            spark
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        show_performance_summary(
            original_partitions,
            aggregation_time,
            temporal_time,
            top_activity_time
        )

        # ====================================================
        # SUCCESS
        # ====================================================

        print()
        print("=" * 75)
        print(
            "SP5 PERFORMANCE VALIDATION COMPLETED SUCCESSFULLY"
        )
        print("=" * 75)

        print(
            f"SP4 input rows          : "
            f"{sp4_rows:,}"
        )

        print(
            f"Original partitions     : "
            f"{original_partitions}"
        )

        print(
            "Execution plan          : INSPECTED"
        )

        print(
            "Grid aggregation        : PASS"
        )

        print(
            "Temporal aggregation    : PASS"
        )

        print(
            "Top activity query      : PASS"
        )

        print(
            "Repartition validation  : PASS"
        )

        print(
            "Coalesce validation     : PASS"
        )

        print(
            "Data quality            : PASS"
        )

        print(
            "Performance validation  : PASS"
        )

        print(
            "Output format           : Parquet"
        )

        print(
            f"Output location         : "
            f"{SP5_OUTPUT_DIR}"
        )

        print(
            "Status                  : SUCCESS"
        )

        print("=" * 75)

        return {
            "activity_df": activity_df,
            "grid_summary": verified_grid_df,
            "temporal_summary": verified_temporal_df
        }

    except Exception as e:

        print()
        print("=" * 75)
        print("SP5 FAILED")
        print("=" * 75)

        print(
            f"Error type: "
            f"{type(e).__name__}"
        )

        print(
            f"Error: {e}"
        )

        raise

    finally:

        if spark is not None:

            try:

                spark.stop()

            except Exception:

                pass


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_sp5()