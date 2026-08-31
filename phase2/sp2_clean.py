import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "data" / "raw"

# SP2 OUTPUT
# SP2 persists the cleaned curated dataset as Parquet.
# SP3 will read this directory.
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "sp2_clean"


# ============================================================================
# HEADER
# ============================================================================

def print_header(title):

    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


# ============================================================================
# SPARK SESSION
# ============================================================================

def create_spark_session():

    print()
    print("Creating Spark session...")

    spark = (
        SparkSession.builder
        .appName("SP2-Milan-Telecom-Cleaning")
        .master("local[*]")

        # --------------------------------------------------------------------
        # MEMORY SETTINGS
        # --------------------------------------------------------------------

        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")

        # Keep the number of shuffle partitions reasonable for local machine
        .config("spark.sql.shuffle.partitions", "16")

        # --------------------------------------------------------------------
        # TIMEZONE
        # --------------------------------------------------------------------

        .config("spark.sql.session.timeZone", "UTC")

        # --------------------------------------------------------------------
        # DEBUG
        # --------------------------------------------------------------------

        .config(
            "spark.sql.debug.maxToStringFields",
            "200"
        )

        # --------------------------------------------------------------------
        # DISABLE SPECULATIVE EXECUTION
        # --------------------------------------------------------------------

        .config(
            "spark.speculation",
            "false"
        )

        # --------------------------------------------------------------------
        # IMPORTANT PARQUET MEMORY FIX
        #
        # The previous error occurred here:
        #
        # DictionaryValuesWriter.writeDouble()
        #
        # High-cardinality DOUBLE values can consume large amounts of
        # Java heap when Parquet dictionary encoding is enabled.
        #
        # Disable dictionary encoding for safer local execution.
        # --------------------------------------------------------------------

        .config(
            "spark.sql.parquet.enableDictionary",
            "false"
        )

        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("Spark session created successfully")
    print(f"Spark version: {spark.version}")

    print()
    print("Spark configuration:")
    print(f"  Driver memory       : {spark.conf.get('spark.driver.memory')}")
    print(f"  Executor memory     : {spark.conf.get('spark.executor.memory')}")
    print(
        f"  Shuffle partitions  : "
        f"{spark.conf.get('spark.sql.shuffle.partitions')}"
    )
    print(
        f"  Parquet dictionary  : "
        f"{spark.conf.get('spark.sql.parquet.enableDictionary')}"
    )

    return spark


# ============================================================================
# INPUT DISCOVERY
# ============================================================================

def discover_input_files():

    print()
    print("[1] Discovering Milan input files...")
    print("-" * 75)

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"Input directory does not exist:\n{INPUT_DIR}"
        )

    files = sorted(
        [
            path
            for path in INPUT_DIR.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() == ".csv"
                and "mi-" in path.name.lower()
            )
        ]
    )

    if not files:

        raise FileNotFoundError(
            f"No Milan CSV files found in:\n{INPUT_DIR}"
        )

    print(f"Input directory: {INPUT_DIR}")
    print(f"Files found: {len(files)}")

    for file in files:

        print(f"  - {file.name}")

    return files


# ============================================================================
# READ INPUT
# ============================================================================

def read_input_data(spark, files):

    print()
    print("[2] Reading input data with Spark...")
    print("-" * 75)

    file_paths = [
        str(path)
        for path in files
    ]

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("mode", "PERMISSIVE")
        .csv(file_paths)
    )

    input_rows = df.count()

    print(f"Input rows    : {input_rows:,}")
    print(f"Input columns : {len(df.columns)}")

    print()
    print("Raw schema:")
    df.printSchema()

    return df, input_rows


# ============================================================================
# STANDARDIZE COLUMN NAMES
# ============================================================================

def standardize_column_name(name):

    name = name.strip().lower()

    mappings = {

        # Timestamp
        "datetime": "timestamp",

        # Grid
        "cellid": "grid_id",
        "cell_id": "grid_id",

        # Country
        "countrycode": "country_code",
        "country_code": "country_code",

        # SMS
        "smsin": "sms_in",
        "smsout": "sms_out",

        # Calls
        "callin": "call_in",
        "callout": "call_out",

        # Internet
        "internet": "internet_activity",
    }

    if name in mappings:

        return mappings[name]

    replacements = {

        " ": "_",
        "-": "_",
        "/": "_",
        ".": "_",
    }

    for old, new in replacements.items():

        name = name.replace(old, new)

    return name


def standardize_columns(df):

    print()
    print("[3] Standardizing Milan dataset column names...")
    print("-" * 75)

    original_columns = df.columns

    print("Original columns:")

    for column in original_columns:

        print(f"  - {column}")

    for old_column in original_columns:

        new_column = standardize_column_name(old_column)

        if old_column != new_column:

            df = df.withColumnRenamed(
                old_column,
                new_column
            )

    print()
    print("Canonical columns:")

    for column in df.columns:

        print(f"  - {column}")

    return df


# ============================================================================
# REQUIRED COLUMN VALIDATION
# ============================================================================

def validate_required_columns(df):

    print()
    print("[4] Validating required columns...")
    print("-" * 75)

    required_columns = [

        "timestamp",
        "grid_id",
        "country_code",

        "sms_in",
        "sms_out",

        "call_in",
        "call_out",

        "internet_activity",
    ]

    missing = [

        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print()
        print("Available columns:")

        for column in df.columns:

            print(f"  - {column}")

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    print("Status: PASS")
    print("All required canonical columns found.")


# ============================================================================
# CAST DATA TYPES
# ============================================================================

def cast_raw_types(df):

    print()
    print("[5] Casting raw fields to required data types...")
    print("-" * 75)

    df = df.withColumn(
        "timestamp",
        F.to_timestamp(
            F.col("timestamp")
        )
    )

    df = df.withColumn(
        "grid_id",
        F.col("grid_id").cast("long")
    )

    df = df.withColumn(
        "country_code",
        F.col("country_code").cast("long")
    )

    activity_columns = [

        "sms_in",
        "sms_out",
        "call_in",
        "call_out",
        "internet_activity",
    ]

    for column in activity_columns:

        df = df.withColumn(
            column,
            F.col(column).cast("double")
        )

    print("Timestamp type   : TIMESTAMP")
    print("Grid ID type      : LONG")
    print("Country code type : LONG")
    print("Activity types    : DOUBLE")

    print()
    print("Status: PASS")

    return df


# ============================================================================
# REJECT INVALID RECORDS
# ============================================================================

def separate_rejected_records(df):

    print()
    print("[6] Identifying records that must be quarantined...")
    print("-" * 75)

    activity_columns = [

        "sms_in",
        "sms_out",
        "call_in",
        "call_out",
        "internet_activity",
    ]

    missing_grid_condition = (
        F.col("grid_id").isNull()
    )

    missing_timestamp_condition = (
        F.col("timestamp").isNull()
    )

    negative_condition = None

    for column in activity_columns:

        condition = (
            F.col(column).isNotNull()
            & (F.col(column) < 0)
        )

        if negative_condition is None:

            negative_condition = condition

        else:

            negative_condition = (
                negative_condition | condition
            )

    rejected_condition = (
        missing_grid_condition
        | missing_timestamp_condition
        | negative_condition
    )

    rejected_df = df.filter(
        rejected_condition
    )

    clean_df = df.filter(
        ~rejected_condition
    )

    missing_grid_count = (
        df.filter(
            missing_grid_condition
        ).count()
    )

    missing_timestamp_count = (
        df.filter(
            missing_timestamp_condition
        ).count()
    )

    negative_count = (
        df.filter(
            negative_condition
        ).count()
    )

    rejected_count = rejected_df.count()

    clean_count = clean_df.count()

    print()
    print("Rejected-record summary:")

    print(
        f"Missing grid_id       : "
        f"{missing_grid_count:,}"
    )

    print(
        f"Missing timestamp     : "
        f"{missing_timestamp_count:,}"
    )

    print(
        f"Negative activity     : "
        f"{negative_count:,}"
    )

    print(
        f"Total rejected rows   : "
        f"{rejected_count:,}"
    )

    print()

    print(
        f"Clean rows            : "
        f"{clean_count:,}"
    )

    print()
    print(
        "Rejected rows are kept separate "
        "from null-handled rows."
    )

    return (
        clean_df,
        rejected_df,
        rejected_count
    )


# ============================================================================
# PROFILE ACTIVITY NULLS
# ============================================================================

def profile_activity_nulls(df):

    print()
    print("[7] Profiling blank/null activity fields...")
    print("-" * 75)

    activity_columns = [

        "sms_in",
        "sms_out",
        "call_in",
        "call_out",
        "internet_activity",
    ]

    null_counts = {}
    total_nulls = 0

    for column in activity_columns:

        count = (
            df
            .filter(
                F.col(column).isNull()
            )
            .count()
        )

        null_counts[column] = count

        total_nulls += count

        print(
            f"{column:<20}: "
            f"{count:,} null values"
        )

    print()
    print(
        f"Total activity nulls: "
        f"{total_nulls:,}"
    )

    return (
        null_counts,
        total_nulls
    )


# ============================================================================
# NULL -> ZERO
# ============================================================================

def handle_activity_nulls(
    df,
    null_counts
):

    print()
    print("[8] Applying curated-layer null-to-zero rule...")
    print("-" * 75)

    activity_columns = [

        "sms_in",
        "sms_out",
        "call_in",
        "call_out",
        "internet_activity",
    ]

    for column in activity_columns:

        df = df.withColumn(
            column,
            F.coalesce(
                F.col(column),
                F.lit(0.0)
            )
        )

    print()
    print("Nulls converted to 0.0:")

    for column, count in null_counts.items():

        print(
            f"  {column:<20}: "
            f"{count:,}"
        )

    print()
    print("Status: PASS")

    return df


# ============================================================================
# DERIVE DATE
# ============================================================================

def derive_date(df):

    print()
    print("[9] Deriving date...")
    print("-" * 75)

    df = df.withColumn(
        "date",
        F.to_date(
            F.col("timestamp")
        )
    )

    stats = (
        df
        .select(
            F.min("date").alias("min_date"),
            F.max("date").alias("max_date")
        )
        .first()
    )

    print(
        f"Minimum date: "
        f"{stats['min_date']}"
    )

    print(
        f"Maximum date: "
        f"{stats['max_date']}"
    )

    return df


# ============================================================================
# DERIVE HOUR
# ============================================================================

def derive_hour(df):

    print()
    print("[10] Deriving hour...")
    print("-" * 75)

    df = df.withColumn(
        "hour",
        F.hour(
            F.col("timestamp")
        )
    )

    hours = [

        row["hour"]

        for row in (
            df
            .select("hour")
            .distinct()
            .orderBy("hour")
            .collect()
        )
    ]

    print(
        f"Hours found: "
        f"{hours}"
    )

    print(
        f"Number of unique hours: "
        f"{len(hours)}"
    )

    expected_hours = list(
        range(24)
    )

    if hours != expected_hours:

        raise ValueError(
            "Expected all 24 hours "
            f"but found: {hours}"
        )

    print(
        "Status: PASS - all 24 hours found"
    )

    return df


# ============================================================================
# DERIVE DAY OF WEEK
# ============================================================================

def derive_day_of_week(df):

    print()
    print("[11] Deriving day of week...")
    print("-" * 75)

    print(
        "Spark convention:"
    )

    print(
        "1=Sunday, 2=Monday, ..., 7=Saturday"
    )

    df = df.withColumn(
        "day_of_week",
        F.dayofweek(
            F.col("timestamp")
        )
    )

    return df


# ============================================================================
# TIMESTAMP CADENCE
# ============================================================================

def validate_hourly_cadence(
    df,
    number_of_files
):

    print()
    print("[12] Validating hourly timestamp cadence...")
    print("-" * 75)

    distinct_timestamps = (
        df
        .select("timestamp")
        .distinct()
        .count()
    )

    expected_timestamps = (
        number_of_files * 24
    )

    print(
        f"Number of input files        : "
        f"{number_of_files}"
    )

    print(
        f"Distinct timestamps          : "
        f"{distinct_timestamps}"
    )

    print(
        f"Expected timestamps (D × 24) : "
        f"{expected_timestamps}"
    )

    if distinct_timestamps != expected_timestamps:

        raise ValueError(
            "Hourly cadence validation failed: "
            f"expected {expected_timestamps:,}, "
            f"found {distinct_timestamps:,}"
        )

    print(
        "Status: PASS - "
        "D × 24 distinct hourly timestamps found."
    )


# ============================================================================
# TOTAL SMS
# ============================================================================

def create_total_sms(df):

    print()
    print("[13] Creating total_sms...")
    print("-" * 75)

    return df.withColumn(
        "total_sms",
        F.col("sms_in")
        + F.col("sms_out")
    )


# ============================================================================
# TOTAL CALLS
# ============================================================================

def create_total_calls(df):

    print()
    print("[14] Creating total_calls...")
    print("-" * 75)

    return df.withColumn(
        "total_calls",
        F.col("call_in")
        + F.col("call_out")
    )


# ============================================================================
# TOTAL ACTIVITY
# ============================================================================

def create_total_activity(df):

    print()
    print("[15] Creating total_activity...")
    print("-" * 75)

    df = df.withColumn(
        "total_activity",
        F.col("total_sms")
        + F.col("total_calls")
        + F.col("internet_activity")
    )

    print(
        "total_activity = total_sms + "
        "total_calls + internet_activity"
    )

    return df


# ============================================================================
# ROUND METRICS
# ============================================================================

def round_metrics(df):

    print()
    print("[16] Rounding activity metrics...")
    print("-" * 75)

    metric_columns = [

        "sms_in",
        "sms_out",
        "call_in",
        "call_out",
        "internet_activity",

        "total_sms",
        "total_calls",
        "total_activity",
    ]

    for column in metric_columns:

        df = df.withColumn(
            column,
            F.round(
                F.col(column).cast("double"),
                4
            )
        )

    print(
        "Activity metrics rounded "
        "to 4 decimal places."
    )

    return df


# ============================================================================
# FINAL SCHEMA
# ============================================================================

def prepare_final_schema(df):

    print()
    print("[17] Preparing final SP2 schema...")
    print("-" * 75)

    final_columns = [

        "timestamp",
        "date",
        "hour",
        "day_of_week",

        "grid_id",
        "country_code",

        "sms_in",
        "sms_out",

        "call_in",
        "call_out",

        "internet_activity",

        "total_sms",
        "total_calls",
        "total_activity",
    ]

    df = df.select(
        *final_columns
    )

    print("Final SP2 columns:")

    for column in df.columns:

        print(f"  - {column}")

    return df


# ============================================================================
# ROW COUNT VALIDATION
# ============================================================================

def validate_row_counts(
    input_rows,
    rejected_rows,
    clean_rows
):

    print()
    print("[18] Validating row counts...")
    print("-" * 75)

    print(
        f"Input rows    : "
        f"{input_rows:,}"
    )

    print(
        f"Rejected rows : "
        f"{rejected_rows:,}"
    )

    print(
        f"Clean rows    : "
        f"{clean_rows:,}"
    )

    if input_rows != (
        rejected_rows
        + clean_rows
    ):

        raise ValueError(
            "Row-count reconciliation failed: "
            f"input={input_rows:,}, "
            f"rejected={rejected_rows:,}, "
            f"clean={clean_rows:,}"
        )

    print()
    print(
        "Row-count reconciliation: PASS"
    )


# ============================================================================
# FINAL NULL VALIDATION
# ============================================================================

def validate_final_nulls(df):

    print()
    print("[19] Final NULL validation...")
    print("-" * 75)

    null_columns = []

    for column in df.columns:

        count = (
            df
            .filter(
                F.col(column).isNull()
            )
            .count()
        )

        print(
            f"{column:<20}: "
            f"{count:,}"
        )

        if count > 0:

            null_columns.append(
                column
            )

    if null_columns:

        raise ValueError(
            "NULL values remain in "
            "curated columns: "
            + ", ".join(null_columns)
        )

    print()
    print("NULL validation: PASS")


# ============================================================================
# NEGATIVE VALIDATION
# ============================================================================

def validate_no_negative_values(df):

    print()
    print("[20] Final negative activity validation...")
    print("-" * 75)

    activity_columns = [

        "sms_in",
        "sms_out",

        "call_in",
        "call_out",

        "internet_activity",
    ]

    for column in activity_columns:

        count = (
            df
            .filter(
                F.col(column) < 0
            )
            .count()
        )

        print(
            f"{column:<20}: "
            f"{count:,} negative values"
        )

        if count > 0:

            raise ValueError(
                f"Negative values remain "
                f"in {column}"
            )

    print()
    print(
        "Negative activity validation: PASS"
    )


# ============================================================================
# SP2 GRAIN VALIDATION
# ============================================================================

def validate_sp2_grain(df):

    print()
    print("[21] Validating SP2 grain...")
    print("-" * 75)

    print(
        "Expected SP2 grain:"
    )

    print(
        "timestamp + grid_id + country_code"
    )

    total_rows = df.count()

    distinct_grain_rows = (
        df
        .select(
            "timestamp",
            "grid_id",
            "country_code"
        )
        .distinct()
        .count()
    )

    duplicate_rows = (
        total_rows
        - distinct_grain_rows
    )

    print(
        f"Total rows                     : "
        f"{total_rows:,}"
    )

    print(
        f"Distinct timestamp/grid/country: "
        f"{distinct_grain_rows:,}"
    )

    print(
        f"Duplicate grain rows           : "
        f"{duplicate_rows:,}"
    )

    if duplicate_rows != 0:

        raise ValueError(
            "SP2 grain validation failed. "
            "Duplicate records exist at "
            "(timestamp, grid_id, country_code)."
        )

    print(
        "SP2 grain validation: PASS"
    )


# ============================================================================
# WRITE SP2 PARQUET OUTPUT
# ============================================================================

def write_sp2_output(df):

    print()
    print("[22] Writing SP2 output as Parquet...")
    print("-" * 75)

    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )

    # ------------------------------------------------------------------------
    # Create parent directory
    # ------------------------------------------------------------------------

    OUTPUT_DIR.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------------------
    # Remove previous SP2 output
    #
    # Spark's overwrite mode will replace the previous Parquet dataset.
    # ------------------------------------------------------------------------

    print()
    print(
        "Writing cleaned dataset..."
    )

    # ------------------------------------------------------------------------
    # Repartition before writing.
    #
    # This prevents a small number of tasks from receiving excessively large
    # amounts of data and reduces memory pressure during Parquet writing.
    # ------------------------------------------------------------------------

    output_df = df.repartition(16)

    (
        output_df
        .write
        .mode("overwrite")
        .option(
            "compression",
            "snappy"
        )
        .option(
            "parquet.enableDictionary",
            "false"
        )
        .parquet(
            str(OUTPUT_DIR)
        )
    )

    print()
    print(
        "SP2 Parquet write: SUCCESS"
    )

    print(
        f"Parquet output path:\n"
        f"{OUTPUT_DIR}"
    )


# ============================================================================
# VERIFY PARQUET OUTPUT
# ============================================================================

def verify_sp2_output(spark):

    print()
    print("[23] Verifying SP2 Parquet output...")
    print("-" * 75)

    if not OUTPUT_DIR.exists():

        raise FileNotFoundError(
            "SP2 output directory was not created:\n"
            f"{OUTPUT_DIR}"
        )

    parquet_files = [

        path

        for path in OUTPUT_DIR.iterdir()

        if path.is_file()
        and path.suffix.lower() == ".parquet"
    ]

    if not parquet_files:

        raise FileNotFoundError(
            "No Parquet files found in SP2 output directory:\n"
            f"{OUTPUT_DIR}"
        )

    print(
        f"Parquet files found: "
        f"{len(parquet_files)}"
    )

    verify_df = (
        spark.read
        .parquet(
            str(OUTPUT_DIR)
        )
    )

    verify_rows = verify_df.count()

    print(
        f"Rows read back from Parquet: "
        f"{verify_rows:,}"
    )

    print()
    print("Verified SP2 Parquet schema:")

    verify_df.printSchema()

    print()
    print(
        "SP2 Parquet verification: PASS"
    )

    return verify_rows


# ============================================================================
# SAMPLE
# ============================================================================

def show_sample(df):

    print()
    print("[24] Sample cleaned records...")
    print("-" * 75)

    df.show(
        10,
        truncate=False
    )


# ============================================================================
# SUMMARY
# ============================================================================

def show_activity_summary(df):

    print()
    print("[25] Activity summary...")
    print("-" * 75)

    summary_columns = [

        "sms_in",
        "sms_out",

        "call_in",
        "call_out",

        "internet_activity",

        "total_sms",
        "total_calls",
        "total_activity",
    ]

    (
        df
        .select(*summary_columns)
        .summary(
            "count",
            "mean",
            "min",
            "50%",
            "max"
        )
        .show(
            truncate=False
        )
    )


# ============================================================================
# CORE SP2 TRANSFORMATION
# ============================================================================

def run_sp2(spark):

    print_header(
        "SP2 - MILAN TELECOM DATA CLEANING & STANDARDIZATION"
    )

    # ------------------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------------------

    files = discover_input_files()

    (
        df,
        input_rows
    ) = read_input_data(
        spark,
        files
    )

    # ------------------------------------------------------------------------
    # STANDARDIZATION
    # ------------------------------------------------------------------------

    df = standardize_columns(df)

    validate_required_columns(df)

    # ------------------------------------------------------------------------
    # TYPE CASTING
    # ------------------------------------------------------------------------

    df = cast_raw_types(df)

    # ------------------------------------------------------------------------
    # INVALID RECORDS
    # ------------------------------------------------------------------------

    (
        clean_df,
        rejected_df,
        rejected_rows
    ) = separate_rejected_records(df)

    # ------------------------------------------------------------------------
    # NULLS
    # ------------------------------------------------------------------------

    (
        null_counts,
        total_nulls
    ) = profile_activity_nulls(
        clean_df
    )

    clean_df = handle_activity_nulls(
        clean_df,
        null_counts
    )

    # ------------------------------------------------------------------------
    # TIME FIELDS
    # ------------------------------------------------------------------------

    clean_df = derive_date(
        clean_df
    )

    clean_df = derive_hour(
        clean_df
    )

    clean_df = derive_day_of_week(
        clean_df
    )

    # ------------------------------------------------------------------------
    # CADENCE
    # ------------------------------------------------------------------------

    validate_hourly_cadence(
        clean_df,
        len(files)
    )

    # ------------------------------------------------------------------------
    # TOTALS
    # ------------------------------------------------------------------------

    clean_df = create_total_sms(
        clean_df
    )

    clean_df = create_total_calls(
        clean_df
    )

    clean_df = create_total_activity(
        clean_df
    )

    clean_df = round_metrics(
        clean_df
    )

    # ------------------------------------------------------------------------
    # FINAL SCHEMA
    # ------------------------------------------------------------------------

    clean_df = prepare_final_schema(
        clean_df
    )

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    clean_rows = clean_df.count()

    validate_row_counts(
        input_rows,
        rejected_rows,
        clean_rows
    )

    validate_final_nulls(
        clean_df
    )

    validate_no_negative_values(
        clean_df
    )

    validate_sp2_grain(
        clean_df
    )

    # ------------------------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------------------------

    show_sample(
        clean_df
    )

    show_activity_summary(
        clean_df
    )

    # ------------------------------------------------------------------------
    # WRITE PARQUET
    # ------------------------------------------------------------------------

    write_sp2_output(
        clean_df
    )

    # ------------------------------------------------------------------------
    # VERIFY PARQUET
    # ------------------------------------------------------------------------

    parquet_rows = verify_sp2_output(
        spark
    )

    # ------------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "SP2 TRANSFORMATION COMPLETED SUCCESSFULLY"
    )
    print("=" * 75)

    print(
        f"Input rows             : "
        f"{input_rows:,}"
    )

    print(
        f"Rejected rows          : "
        f"{rejected_rows:,}"
    )

    print(
        f"Clean rows             : "
        f"{clean_rows:,}"
    )

    print(
        f"Activity nulls handled : "
        f"{total_nulls:,}"
    )

    print(
        f"Input files            : "
        f"{len(files)}"
    )

    print(
        f"Expected timestamps    : "
        f"{len(files) * 24}"
    )

    print(
        f"Parquet rows verified  : "
        f"{parquet_rows:,}"
    )

    print(
        "Output grain           : "
        "timestamp + grid_id + country_code"
    )

    print(
        "Output format          : Parquet"
    )

    print(
        f"Output location        : "
        f"{OUTPUT_DIR}"
    )

    print(
        "Next stage             : SP3"
    )

    print(
        "SP3 input              : "
        "SP2 Parquet output"
    )

    print(
        "Status                 : SUCCESS"
    )

    print("=" * 75)

    # ------------------------------------------------------------------------
    # RETURN SP2 DATAFRAME
    # ------------------------------------------------------------------------

    return clean_df


# ============================================================================
# STANDALONE SP2 EXECUTION
# ============================================================================

def main():

    spark = None

    try:

        spark = create_spark_session()

        clean_df = run_sp2(
            spark
        )

        print()
        print(
            "SP2 standalone execution completed."
        )

        print(
            "SP2 Parquet output is ready for SP3."
        )

        print(
            f"SP2 output:\n"
            f"{OUTPUT_DIR}"
        )

    except Exception as exc:

        print()
        print("=" * 75)
        print(
            "SP2 TRANSFORMATION FAILED"
        )
        print("=" * 75)

        print(
            f"Error type: "
            f"{type(exc).__name__}"
        )

        print(
            f"Error: "
            f"{exc}"
        )

        sys.exit(1)

    finally:

        if spark is not None:

            try:

                spark.stop()

            except Exception:

                pass


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()