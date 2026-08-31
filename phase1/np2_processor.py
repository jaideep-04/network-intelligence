import logging
from pathlib import Path

import pandas as pd


class UsageProcessor:

    # -------------------------------------------------------
    # Configuration
    # -------------------------------------------------------

    ACTIVITY_COLUMNS = [
        "sms_in",
        "sms_out",
        "call_in",
        "call_out",
        "internet_activity"
    ]

    RAW_KEY_COLUMNS = [
        "timestamp",
        "grid_id",
        "country_code"
    ]

    CANONICAL_RENAME = {
        "datetime": "timestamp",
        "CellID": "grid_id",
        "countrycode": "country_code",
        "smsin": "sms_in",
        "smsout": "sms_out",
        "callin": "call_in",
        "callout": "call_out",
        "internet": "internet_activity"
    }

    # -------------------------------------------------------
    # Constructor
    # -------------------------------------------------------

    def __init__(self, input_path, output_dir="../outputs"):

        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.raw_df = None
        self.curated_df = None
        self.grid_hour_df = None

        self.input_rows = 0
        self.rejected_rows = 0
        self.nulls_handled = 0

        self.logger = logging.getLogger(
            "UsageProcessor"
        )

        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:

            handler = logging.FileHandler(
                self.output_dir / "usage_processor.log"
            )

            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )

            handler.setFormatter(formatter)

            self.logger.addHandler(handler)

    # -------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------

    def load_data(self):

        if not self.input_path.exists():

            raise FileNotFoundError(
                f"Input file not found: "
                f"{self.input_path}"
            )

        self.logger.info(
            "Loading file: %s",
            self.input_path
        )

        df = pd.read_csv(self.input_path)

        self.input_rows = len(df)

        self.logger.info(
            "Input rows: %d",
            self.input_rows
        )

        # Validate expected raw columns

        missing_columns = set(
            self.CANONICAL_RENAME.keys()
        ) - set(df.columns)

        if missing_columns:

            raise ValueError(
                "Missing expected columns: "
                f"{sorted(missing_columns)}"
            )

        # Preserve original raw dataframe

        self.raw_df = df.copy()

        # Create canonical dataframe

        self.curated_df = df.rename(
            columns=self.CANONICAL_RENAME
        ).copy()

        return self.curated_df

    # -------------------------------------------------------
    # 2. CLEAN DATA
    # -------------------------------------------------------

    def clean_data(self):

        if self.curated_df is None:

            raise RuntimeError(
                "load_data() must be called first"
            )

        df = self.curated_df.copy()

        # -----------------------------------------------
        # Convert timestamp
        # -----------------------------------------------

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        # -----------------------------------------------
        # Convert activity columns to numeric
        # -----------------------------------------------

        for column in self.ACTIVITY_COLUMNS:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # -----------------------------------------------
        # Identify invalid key fields
        # -----------------------------------------------

        invalid_key_mask = (
            df["timestamp"].isna()
            | df["grid_id"].isna()
        )

        # -----------------------------------------------
        # Identify negative activity
        # -----------------------------------------------

        negative_mask = pd.Series(
            False,
            index=df.index
        )

        for column in self.ACTIVITY_COLUMNS:

            negative_mask |= (
                df[column].notna()
                & (df[column] < 0)
            )

        # -----------------------------------------------
        # Reject invalid rows
        # -----------------------------------------------

        reject_mask = (
            invalid_key_mask
            | negative_mask
        )

        self.rejected_rows = int(
            reject_mask.sum()
        )

        if self.rejected_rows > 0:

            self.logger.warning(
                "Rejected rows: %d",
                self.rejected_rows
            )

        df = df.loc[
            ~reject_mask
        ].copy()

        # -----------------------------------------------
        # Count activity nulls BEFORE filling
        # -----------------------------------------------

        nulls_before = (
            df[self.ACTIVITY_COLUMNS]
            .isna()
            .sum()
            .sum()
        )

        self.nulls_handled = int(
            nulls_before
        )

        # -----------------------------------------------
        # Curated null-to-zero rule
        # -----------------------------------------------

        df[self.ACTIVITY_COLUMNS] = (
            df[self.ACTIVITY_COLUMNS]
            .fillna(0)
        )

        self.logger.info(
            "Null activity values handled: %d",
            self.nulls_handled
        )

        self.curated_df = df

        return df

    # -------------------------------------------------------
    # 3. DERIVE TIME FEATURES
    # -------------------------------------------------------

    def derive_time_features(self):

        if self.curated_df is None:

            raise RuntimeError(
                "clean_data() must be called first"
            )

        df = self.curated_df.copy()

        df["date"] = (
            df["timestamp"]
            .dt.date
        )

        df["hour"] = (
            df["timestamp"]
            .dt.hour
        )

        df["day_of_week"] = (
            df["timestamp"]
            .dt.day_name()
        )

        self.curated_df = df

        self.logger.info(
            "Derived date, hour and day_of_week"
        )

        return df

    # -------------------------------------------------------
    # 4. AGGREGATE TO GRID / TIME
    # -------------------------------------------------------

    def aggregate_to_grid_time(self):

        if self.curated_df is None:

            raise RuntimeError(
                "derive_time_features() "
                "must be called first"
            )

        df = self.curated_df.copy()

        input_rows = len(df)

        # -----------------------------------------------
        # Aggregate country-code rows
        # -----------------------------------------------

        grouped = (
            df.groupby(
                [
                    "timestamp",
                    "grid_id"
                ],
                as_index=False
            )[
                self.ACTIVITY_COLUMNS
            ]
            .sum()
        )

        # -----------------------------------------------
        # Validate grain
        # -----------------------------------------------

        duplicate_count = grouped.duplicated(
            subset=[
                "timestamp",
                "grid_id"
            ]
        ).sum()

        if duplicate_count != 0:

            raise ValueError(
                "Grid/hour aggregation failed. "
                f"Found {duplicate_count} "
                "duplicate grid/hour records."
            )

        output_rows = len(grouped)

        if output_rows >= input_rows:

            raise ValueError(
                "Aggregation did not reduce "
                "the number of rows."
            )

        # -----------------------------------------------
        # Re-add time features
        # -----------------------------------------------

        grouped["date"] = (
            grouped["timestamp"]
            .dt.date
        )

        grouped["hour"] = (
            grouped["timestamp"]
            .dt.hour
        )

        grouped["day_of_week"] = (
            grouped["timestamp"]
            .dt.day_name()
        )

        self.grid_hour_df = grouped

        self.logger.info(
            "Grid/hour aggregation complete | "
            "input rows=%d | output rows=%d",
            input_rows,
            output_rows
        )

        return grouped

    # -------------------------------------------------------
    # 5. DERIVE ACTIVITY FEATURES
    # -------------------------------------------------------

    def derive_activity_features(self):

        if self.grid_hour_df is None:

            raise RuntimeError(
                "aggregate_to_grid_time() "
                "must be called first"
            )

        df = self.grid_hour_df.copy()

        df["total_sms"] = (
            df["sms_in"]
            + df["sms_out"]
        )

        df["total_calls"] = (
            df["call_in"]
            + df["call_out"]
        )

        df["total_activity"] = (
            df["total_sms"]
            + df["total_calls"]
            + df["internet_activity"]
        )

        self.grid_hour_df = df

        self.logger.info(
            "Derived total_sms, total_calls "
            "and total_activity"
        )

        return df

    # -------------------------------------------------------
    # 6. COMPUTE KPIs
    # -------------------------------------------------------

    def compute_kpis(self):

        if self.grid_hour_df is None:

            raise RuntimeError(
                "derive_activity_features() "
                "must be called first"
            )

        df = self.grid_hour_df.copy()

        # -----------------------------------------------
        # Daily summary
        # -----------------------------------------------

        daily_summary = (
            df.groupby(
                "date",
                as_index=False
            )
            .agg(
                total_sms=(
                    "total_sms",
                    "sum"
                ),
                total_calls=(
                    "total_calls",
                    "sum"
                ),
                internet_activity=(
                    "internet_activity",
                    "sum"
                ),
                total_activity=(
                    "total_activity",
                    "sum"
                )
            )
        )

        # -----------------------------------------------
        # Grid summary
        # -----------------------------------------------

        grid_summary = (
            df.groupby(
                "grid_id",
                as_index=False
            )
            .agg(
                total_sms=(
                    "total_sms",
                    "sum"
                ),
                total_calls=(
                    "total_calls",
                    "sum"
                ),
                internet_activity=(
                    "internet_activity",
                    "sum"
                ),
                total_activity=(
                    "total_activity",
                    "sum"
                )
            )
        )

        # -----------------------------------------------
        # Internet share
        # -----------------------------------------------

        grid_summary["internet_share"] = (
            grid_summary["internet_activity"]
            / grid_summary["total_activity"]
            .replace(0, pd.NA)
        )

        return {
            "daily_summary": daily_summary,
            "grid_summary": grid_summary
        }

    # -------------------------------------------------------
    # 7. EXPORT SUMMARY
    # -------------------------------------------------------

    def export_summary(self):

        summaries = self.compute_kpis()

        daily_summary = summaries[
            "daily_summary"
        ]

        grid_summary = summaries[
            "grid_summary"
        ]

        daily_path = (
            self.output_dir
            / "daily_summary.csv"
        )

        grid_path = (
            self.output_dir
            / "grid_summary.csv"
        )

        daily_summary.to_csv(
            daily_path,
            index=False
        )

        grid_summary.to_csv(
            grid_path,
            index=False
        )

        self.logger.info(
            "Input rows: %d",
            self.input_rows
        )

        self.logger.info(
            "Rows rejected: %d",
            self.rejected_rows
        )

        self.logger.info(
            "Nulls handled: %d",
            self.nulls_handled
        )

        self.logger.info(
            "Output rows: %d",
            len(self.grid_hour_df)
        )
        grid_hour_path = (
            self.output_dir
            / "grid_hour_activity.csv"
        )

        self.grid_hour_df.to_csv(
            grid_hour_path,
            index=False
        )

        self.logger.info(
            "Grid/hour activity exported: %s",
            grid_hour_path
        )

        return daily_summary, grid_summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    processor = UsageProcessor(
        input_path=(
            "C:\\Users\\jaideep.d\\network-intelligence\\data\\raw\\sms-call-internet-mi-2013-11-01.csv"
        ),
        output_dir="../outputs"
    )

    processor.load_data()

    processor.clean_data()

    processor.derive_time_features()

    processor.aggregate_to_grid_time()

    processor.derive_activity_features()

    daily_summary, grid_summary = (
        processor.export_summary()
    )

    print("=" * 75)
    print("NP2 - USAGE PROCESSOR COMPLETE")
    print("=" * 75)

    print(
        f"Input rows      : "
        f"{processor.input_rows:,}"
    )

    print(
        f"Rejected rows   : "
        f"{processor.rejected_rows:,}"
    )

    print(
        f"Nulls handled   : "
        f"{processor.nulls_handled:,}"
    )

    print(
        f"Grid/hour rows  : "
        f"{len(processor.grid_hour_df):,}"
    )

    print(
        f"Daily summary   : "
        f"{len(daily_summary):,} rows"
    )

    print(
        f"Grid summary    : "
        f"{len(grid_summary):,} rows"
    )

    print("=" * 75)

    