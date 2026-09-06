from pathlib import Path
import sqlite3
import ast

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone

from pydantic import BaseModel, Field
import json

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "warehouse" / "network_analytics.db"

app = FastAPI(
    title="Network Intelligence API",
    version="1.0.0",
    description="REST API for telecom network intelligence analytics",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Network analytics database is unavailable",
        )

    try:
        connection = sqlite3.connect(
            f"file:{DB_PATH.as_posix()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        return connection

    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to connect to analytics database: {exc}",
        )


# ============================================================
# RESPONSE MODELS
# ============================================================

class NetworkSummaryResponse(BaseModel):
    total_activity: float
    active_grids: int
    peak_hour: int
    top_grid: int
    as_of: str


class GridActivityPoint(BaseModel):
    timestamp: str
    date: str
    hour: int
    sms_in: float
    sms_out: float
    call_in: float
    call_out: float
    internet_activity: float
    total_activity: float


class GridActivityResponse(BaseModel):
    grid_id: int
    as_of: str
    points: list[GridActivityPoint]


class NetworkAlert(BaseModel):
    alert_id: int
    timestamp: str
    grid_id: int
    alert_type: str
    severity: str
    metric: str
    value: float
    threshold: float
    status: str
    reason: str
    risk_score: float | None = None


class NetworkAlertsResponse(BaseModel):
    as_of: str
    count: int
    alerts: list[NetworkAlert]
class GridFeatureResponse(BaseModel):
    grid_id: int
    feature_timestamp: str
    avg_activity: float
    activity_growth: float
    active_hours: int
    peak_ratio: float
    variability: float
    internet_share: float
    data_quality_status: str
    freshness_hours: float
class PredictRiskRequest(BaseModel):
    grid_id: int = Field(..., ge=1, le=10000)
    feature_timestamp: str

    avg_activity: float = Field(..., ge=0)
    activity_growth: float
    active_hours: int = Field(..., ge=0, le=24)
    peak_ratio: float = Field(..., ge=0)
    variability: float = Field(..., ge=0)
    internet_share: float = Field(..., ge=0, le=1)


class PredictRiskResponse(BaseModel):
    risk_score: float
    risk_level: str
    model_version: str
    explanation_note: str
class PipelineStatusResponse(BaseModel):
    run_id: str
    run_timestamp: str
    status: str
    healthy: bool
    reasons: list[str]
    tasks: dict[str, str]
    rows_in: int
    rows_rejected: int
    nulls_handled: int | None
    rows_published: int
    as_of: str
    analytics_freshness_hours: float
    dim_time_rows: int
    dim_grid_rows: int
    fact_rows: int
    orphan_time_keys: int
    orphan_grid_keys: int


class GridLocationResponse(BaseModel):
    grid_id: int
    polygon_reference: str
    centroid_latitude: float
    centroid_longitude: float


# ============================================================
# HELPERS
# ============================================================
def load_pipeline_status_record():
    status_path = (
        BASE_DIR
        / "logs"
        / "pipeline_status"
        / "latest_pipeline_status.json"
    )

    if not status_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Pipeline status record is unavailable",
        )

    try:
        with status_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read pipeline status: {exc}",
        )
def get_effective_as_of(
    connection: sqlite3.Connection,
    requested_as_of: str | None,
) -> str:

    if requested_as_of:

        row = connection.execute(
            """
            SELECT timestamp
            FROM dim_time
            WHERE timestamp = ?
            LIMIT 1
            """,
            (requested_as_of,),
        ).fetchone()

        if row is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid as_of timestamp: {requested_as_of}",
            )

        return row["timestamp"]

    row = connection.execute(
        """
        SELECT MAX(timestamp) AS as_of
        FROM dim_time
        """
    ).fetchone()

    if row is None or row["as_of"] is None:
        raise HTTPException(
            status_code=500,
            detail="Analytics warehouse contains no timestamps",
        )

    return row["as_of"]


def get_alert_thresholds(connection):
    """
    Calculate the same NP3 thresholds from the stored
    hourly analytics data.

    NP3:
        high activity   = 95th percentile
        high internet   = 95th percentile
        low activity    = 5th percentile
        spike multiplier = 2.0
    """

    values = connection.execute(
        """
        SELECT
            total_activity,
            internet_activity
        FROM fact_network_activity
        """
    ).fetchall()

    if not values:
        raise HTTPException(
            status_code=500,
            detail="No network activity data available",
        )

    total_values = sorted(
        float(row["total_activity"])
        for row in values
    )

    internet_values = sorted(
        float(row["internet_activity"])
        for row in values
    )

    def percentile(sorted_values, percentile):
        if not sorted_values:
            return 0.0

        position = (
            (len(sorted_values) - 1)
            * percentile
        )

        lower = int(position)
        upper = min(
            lower + 1,
            len(sorted_values) - 1,
        )

        fraction = position - lower

        return (
            sorted_values[lower]
            + (
                sorted_values[upper]
                - sorted_values[lower]
            )
            * fraction
        )

    return {
        "high_activity": percentile(
            total_values,
            0.95,
        ),
        "high_internet": percentile(
            internet_values,
            0.95,
        ),
        "low_activity": percentile(
            total_values,
            0.05,
        ),
        "spike_multiplier": 2.0,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "network-intelligence-api",
    }


# ============================================================
# API1 - NETWORK SUMMARY
# ============================================================

@app.get(
    "/network/summary",
    response_model=NetworkSummaryResponse,
)
def network_summary(
    as_of: str | None = Query(
        default=None,
        description="Optional effective analytics timestamp",
    ),
):

    connection = get_connection()

    try:

        effective_as_of = get_effective_as_of(
            connection,
            as_of,
        )

        summary = connection.execute(
            """
            SELECT
                COALESCE(
                    SUM(f.total_activity),
                    0
                ) AS total_activity,

                COUNT(
                    CASE
                        WHEN f.total_activity > 0
                        THEN 1
                    END
                ) AS active_grids

            FROM fact_network_activity f

            JOIN dim_time t
                ON f.time_id = t.time_id

            WHERE t.timestamp = ?
            """,
            (effective_as_of,),
        ).fetchone()

        peak = connection.execute(
            """
            SELECT
                t.timestamp,
                t.hour AS peak_hour,
                SUM(f.total_activity) AS activity

            FROM fact_network_activity f

            JOIN dim_time t
                ON f.time_id = t.time_id

            GROUP BY
                t.timestamp,
                t.hour

            ORDER BY
                activity DESC,
                t.timestamp ASC,
                t.hour ASC

            LIMIT 1
            """
        ).fetchone()

        top_grid = connection.execute(
            """
            SELECT
                f.grid_id,
                SUM(f.total_activity) AS activity

            FROM fact_network_activity f

            JOIN dim_time t
                ON f.time_id = t.time_id

            WHERE t.timestamp = ?

            GROUP BY f.grid_id

            ORDER BY
                activity DESC,
                f.grid_id ASC

            LIMIT 1
            """,
            (effective_as_of,),
        ).fetchone()

        if (
            summary is None
            or peak is None
            or top_grid is None
        ):
            raise HTTPException(
                status_code=500,
                detail="Unable to calculate network summary",
            )

        return NetworkSummaryResponse(
            total_activity=float(
                summary["total_activity"]
            ),
            active_grids=int(
                summary["active_grids"]
            ),
            peak_hour=int(
                peak["peak_hour"]
            ),
            top_grid=int(
                top_grid["grid_id"]
            ),
            as_of=effective_as_of,
        )

    except HTTPException:
        raise

    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Database query failed: {exc}",
        )

    finally:
        connection.close()


# ============================================================
# API2 - GRID ACTIVITY
# ============================================================

@app.get(
    "/network/grid/{grid_id}",
    response_model=GridActivityResponse,
)
def network_grid(
    grid_id: int,

    date: str | None = Query(
        default=None,
        description="Optional date filter in YYYY-MM-DD format",
    ),

    hour: int | None = Query(
        default=None,
        ge=0,
        le=23,
        description="Optional hour filter from 0 to 23",
    ),

    as_of: str | None = Query(
        default=None,
        description="Optional effective analytics timestamp",
    ),
):

    if grid_id < 1 or grid_id > 10000:
        raise HTTPException(
            status_code=404,
            detail=f"Grid {grid_id} not found",
        )

    connection = get_connection()

    try:

        grid = connection.execute(
            """
            SELECT grid_id
            FROM dim_grid
            WHERE grid_id = ?
            LIMIT 1
            """,
            (grid_id,),
        ).fetchone()

        if grid is None:
            raise HTTPException(
                status_code=404,
                detail=f"Grid {grid_id} not found",
            )

        effective_as_of = get_effective_as_of(
            connection,
            as_of,
        )

        conditions = [
            "f.grid_id = ?",
            "t.timestamp <= ?",
        ]

        parameters = [
            grid_id,
            effective_as_of,
        ]

        if date is not None:
            conditions.append("t.date = ?")
            parameters.append(date)

        if hour is not None:
            conditions.append("t.hour = ?")
            parameters.append(hour)

        where_clause = " AND ".join(
            conditions
        )

        query = f"""
            SELECT
                t.timestamp,
                t.date,
                t.hour,
                f.sms_in,
                f.sms_out,
                f.call_in,
                f.call_out,
                f.internet_activity,
                f.total_activity

            FROM fact_network_activity f

            JOIN dim_time t
                ON f.time_id = t.time_id

            WHERE {where_clause}

            ORDER BY t.timestamp DESC
        """

        rows = connection.execute(
            query,
            tuple(parameters),
        ).fetchall()

        if date is None and hour is None:
            rows = rows[:24]

        rows = list(reversed(rows))

        points = [
            GridActivityPoint(
                timestamp=row["timestamp"],
                date=row["date"],
                hour=int(row["hour"]),
                sms_in=float(row["sms_in"]),
                sms_out=float(row["sms_out"]),
                call_in=float(row["call_in"]),
                call_out=float(row["call_out"]),
                internet_activity=float(
                    row["internet_activity"]
                ),
                total_activity=float(
                    row["total_activity"]
                ),
            )
            for row in rows
        ]

        return GridActivityResponse(
            grid_id=grid_id,
            as_of=effective_as_of,
            points=points,
        )

    except HTTPException:
        raise

    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Database query failed: {exc}",
        )

    finally:
        connection.close()


# ============================================================
# API3 - HOTSPOTS
# ============================================================

@app.get(
    "/network/hotspots",
    response_model=NetworkAlertsResponse,
)
def network_hotspots(
    limit: int = Query(
        default=10,
        ge=1,
        le=1000,
    ),

    severity: str | None = Query(
        default=None,
    ),

    as_of: str | None = Query(
        default=None,
    ),
):

    connection = get_connection()

    try:

        effective_as_of = get_effective_as_of(
            connection,
            as_of,
        )

        thresholds = get_alert_thresholds(
            connection
        )

        rows = connection.execute(
            """
            SELECT
                t.timestamp,
                f.grid_id,
                f.internet_activity,
                f.total_activity

            FROM fact_network_activity f

            JOIN dim_time t
                ON f.time_id = t.time_id

            WHERE t.timestamp = ?

            ORDER BY
                f.total_activity DESC,
                f.grid_id ASC
            """,
            (effective_as_of,),
        ).fetchall()

        alerts = []

        for row in rows:

            if (
                row["total_activity"]
                > thresholds["high_activity"]
            ):

                alerts.append(
                    NetworkAlert(
                        alert_id=len(alerts) + 1,
                        timestamp=row["timestamp"],
                        grid_id=int(row["grid_id"]),
                        alert_type="HIGH_ACTIVITY",
                        severity="HIGH",
                        metric="total_activity",
                        value=float(
                            row["total_activity"]
                        ),
                        threshold=float(
                            thresholds["high_activity"]
                        ),
                        status="ACTIVE",
                        reason=(
                            "Total activity exceeds "
                            "the 95th percentile threshold"
                        ),
                        risk_score=None,
                    )
                )

        if severity:
            severity_upper = severity.upper()

            alerts = [
                alert
                for alert in alerts
                if alert.severity == severity_upper
            ]

        alerts = alerts[:limit]

        return NetworkAlertsResponse(
            as_of=effective_as_of,
            count=len(alerts),
            alerts=alerts,
        )

    except HTTPException:
        raise

    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Database query failed: {exc}",
        )

    finally:
        connection.close()


# ============================================================
# API3 - ALERTS
# ============================================================

@app.get(
    "/network/alerts",
    response_model=NetworkAlertsResponse,
)
def network_alerts(
    limit: int = Query(
        default=10,
        ge=1,
        le=1000,
    ),

    severity: str | None = Query(
        default=None,
    ),

    as_of: str | None = Query(
        default=None,
    ),
):

    connection = get_connection()

    try:

        effective_as_of = get_effective_as_of(
            connection,
            as_of,
        )

        thresholds = get_alert_thresholds(
            connection
        )

        rows = connection.execute(
            """
            SELECT
                t.timestamp,
                t.date,
                t.hour,
                f.grid_id,
                f.internet_activity,
                f.total_activity

            FROM fact_network_activity f

            JOIN dim_time t
                ON f.time_id = t.time_id

            WHERE t.timestamp <= ?

            ORDER BY
                t.timestamp ASC,
                f.grid_id ASC
            """,
            (effective_as_of,),
        ).fetchall()

        alerts = []

        previous_activity = {}

        for row in rows:

            timestamp = row["timestamp"]
            grid_id = int(row["grid_id"])

            total_activity = float(
                row["total_activity"]
            )

            internet_activity = float(
                row["internet_activity"]
            )

            # ----------------------------------------------
            # RULE 1 - HIGH ACTIVITY
            # ----------------------------------------------

            if (
                total_activity
                > thresholds["high_activity"]
            ):

                alerts.append(
                    NetworkAlert(
                        alert_id=0,
                        timestamp=timestamp,
                        grid_id=grid_id,
                        alert_type="HIGH_ACTIVITY",
                        severity="HIGH",
                        metric="total_activity",
                        value=total_activity,
                        threshold=float(
                            thresholds["high_activity"]
                        ),
                        status="ACTIVE",
                        reason=(
                            "Total activity exceeds "
                            "the 95th percentile threshold"
                        ),
                    )
                )

            # ----------------------------------------------
            # RULE 2 - HIGH INTERNET
            # ----------------------------------------------

            if (
                internet_activity
                > thresholds["high_internet"]
            ):

                alerts.append(
                    NetworkAlert(
                        alert_id=0,
                        timestamp=timestamp,
                        grid_id=grid_id,
                        alert_type="HIGH_INTERNET",
                        severity="MEDIUM",
                        metric="internet_activity",
                        value=internet_activity,
                        threshold=float(
                            thresholds["high_internet"]
                        ),
                        status="ACTIVE",
                        reason=(
                            "Internet activity exceeds "
                            "the 95th percentile threshold"
                        ),
                    )
                )

            # ----------------------------------------------
            # RULE 3 - SUDDEN SPIKE
            # ----------------------------------------------

            previous = previous_activity.get(
                grid_id
            )

            if (
                previous is not None
                and total_activity
                > previous * thresholds["spike_multiplier"]
            ):

                alerts.append(
                    NetworkAlert(
                        alert_id=0,
                        timestamp=timestamp,
                        grid_id=grid_id,
                        alert_type="SUDDEN_SPIKE",
                        severity="HIGH",
                        metric="total_activity",
                        value=total_activity,
                        threshold=(
                            previous
                            * thresholds["spike_multiplier"]
                        ),
                        status="ACTIVE",
                        reason=(
                            "Total activity is more than "
                            "2 times the previous hourly value"
                        ),
                    )
                )

            # ----------------------------------------------
            # RULE 4 - LOW ACTIVITY
            # ----------------------------------------------

            if (
                total_activity
                < thresholds["low_activity"]
            ):

                alerts.append(
                    NetworkAlert(
                        alert_id=0,
                        timestamp=timestamp,
                        grid_id=grid_id,
                        alert_type="LOW_ACTIVITY",
                        severity="LOW",
                        metric="total_activity",
                        value=total_activity,
                        threshold=float(
                            thresholds["low_activity"]
                        ),
                        status="ACTIVE",
                        reason=(
                            "Total activity is below "
                            "the 5th percentile threshold"
                        ),
                    )
                )

            previous_activity[grid_id] = (
                total_activity
            )

        # ----------------------------------------------------
        # Deterministic ordering matching NP3
        # ----------------------------------------------------

        severity_order = {
            "HIGH": 0,
            "MEDIUM": 1,
            "LOW": 2,
        }

        alerts.sort(
            key=lambda alert: (
                alert.timestamp,
                alert.grid_id,
                alert.alert_type,
            )
        )

        if severity:
            severity_upper = severity.upper()

            alerts = [
                alert
                for alert in alerts
                if alert.severity == severity_upper
            ]

        alerts = alerts[:limit]

        # Assign stable IDs after sorting/filtering
        for index, alert in enumerate(
            alerts,
            start=1,
        ):
            alert.alert_id = index

        return NetworkAlertsResponse(
            as_of=effective_as_of,
            count=len(alerts),
            alerts=alerts,
        )

    except HTTPException:
        raise

    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Database query failed: {exc}",
        )

    finally:
        connection.close()
# ============================================================
# API4 - GRID ML FEATURES
# ============================================================

@app.get(
    "/network/grid/{grid_id}/features",
    response_model=GridFeatureResponse,
)
def network_grid_features(grid_id: int):

    if grid_id < 1 or grid_id > 10000:
        raise HTTPException(
            status_code=404,
            detail=f"Grid {grid_id} not found",
        )

    connection = get_connection()

    try:

        # ----------------------------------------------------
        # Verify that the grid exists
        # ----------------------------------------------------

        grid = connection.execute(
            """
            SELECT grid_id
            FROM dim_grid
            WHERE grid_id = ?
            LIMIT 1
            """,
            (grid_id,),
        ).fetchone()

        if grid is None:
            raise HTTPException(
                status_code=404,
                detail=f"Grid {grid_id} not found",
            )

        # ----------------------------------------------------
        # Get the latest stored feature row.
        #
        # IMPORTANT:
        # FastAPI does NOT calculate any ML features.
        # All six values come directly from network_features.
        # ----------------------------------------------------

        feature = connection.execute(
            """
            SELECT
                grid_id,
                feature_timestamp,
                avg_activity,
                activity_growth,
                active_hours,
                peak_ratio,
                variability,
                internet_share

            FROM network_features

            WHERE grid_id = ?

            ORDER BY feature_timestamp DESC

            LIMIT 1
            """,
            (grid_id,),
        ).fetchone()

        if feature is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No stored ML features available "
                    f"for grid {grid_id}"
                ),
            )

        # ----------------------------------------------------
        # Get warehouse AS_OF.
        #
        # This is used only to report feature freshness.
        # It does not modify or calculate ML features.
        # ----------------------------------------------------

        as_of_row = connection.execute(
            """
            SELECT MAX(timestamp) AS as_of
            FROM dim_time
            """
        ).fetchone()

        if (
            as_of_row is None
            or as_of_row["as_of"] is None
        ):
            raise HTTPException(
                status_code=500,
                detail=(
                    "Analytics warehouse contains "
                    "no timestamps"
                ),
            )

        as_of_timestamp = as_of_row["as_of"]
        feature_timestamp = feature["feature_timestamp"]

        # ----------------------------------------------------
        # Freshness calculation
        # ----------------------------------------------------

        try:
            from datetime import datetime

            as_of_dt = datetime.fromisoformat(
                as_of_timestamp
            )

            feature_dt = datetime.fromisoformat(
                feature_timestamp
            )

            freshness_hours = (
                as_of_dt - feature_dt
            ).total_seconds() / 3600.0

        except ValueError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Unable to calculate feature freshness: "
                    f"{exc}"
                ),
            )

        if freshness_hours < 0:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Feature timestamp is newer than "
                    "the warehouse AS_OF timestamp"
                ),
            )

        # ----------------------------------------------------
        # Data-quality status
        # ----------------------------------------------------

        data_quality_status = "VALID"

        # ----------------------------------------------------
        # Return stored ML2 values exactly as persisted.
        # ----------------------------------------------------

        return GridFeatureResponse(
            grid_id=int(feature["grid_id"]),
            feature_timestamp=feature[
                "feature_timestamp"
            ],
            avg_activity=float(
                feature["avg_activity"]
            ),
            activity_growth=float(
                feature["activity_growth"]
            ),
            active_hours=int(
                feature["active_hours"]
            ),
            peak_ratio=float(
                feature["peak_ratio"]
            ),
            variability=float(
                feature["variability"]
            ),
            internet_share=float(
                feature["internet_share"]
            ),
            data_quality_status=data_quality_status,
            freshness_hours=float(
                freshness_hours
            ),
        )

    except HTTPException:
        raise

    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Database query failed: {exc}",
        )

    finally:
        connection.close()
@app.post(
    "/network/predict-risk",
    response_model=PredictRiskResponse,
)
def predict_risk(request: PredictRiskRequest):

    return PredictRiskResponse(
        risk_score=0.0,
        risk_level="STUB",
        model_version="stub-v1",
        explanation_note=(
            "Prediction implementation is currently a stub. "
            "The trained ML model will replace this implementation "
            "in ML5 without changing the API contract."
        ),
    )
@app.get(
    "/pipeline/status",
    response_model=PipelineStatusResponse,
)
def pipeline_status():

    status_data = load_pipeline_status_record()

    required_fields = [
        "run_id",
        "run_timestamp",
        "status",
        "tasks",
        "rows_in",
        "rows_rejected",
        "rows_published",
        "dim_time_rows",
        "dim_grid_rows",
        "fact_rows",
        "orphan_time_keys",
        "orphan_grid_keys",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in status_data
    ]

    if missing_fields:
        raise HTTPException(
            status_code=500,
            detail=(
                "Pipeline status record is missing fields: "
                + ", ".join(missing_fields)
            ),
        )

    # --------------------------------------------------------
    # Determine overall health from the machine-readable record
    # --------------------------------------------------------
    reasons = []

    overall_status = str(
        status_data["status"]
    ).upper()

    if overall_status != "SUCCESS":
        reasons.append(
            f"Pipeline status is {overall_status}"
        )

    tasks = {
        str(key): str(value)
        for key, value in status_data["tasks"].items()
    }

    # Notify is a downstream notification task and can remain
    # PENDING without making the analytics pipeline unhealthy.
    for task_name, task_status in tasks.items():
        if task_name == "notify":
            continue

        if task_status.upper() != "SUCCESS":
            reasons.append(
                f"Task '{task_name}' status is {task_status}"
            )

    healthy = len(reasons) == 0

    # --------------------------------------------------------
    # Read canonical AS_OF from the warehouse
    # --------------------------------------------------------
    connection = get_connection()

    try:
        warehouse_as_of = connection.execute(
            """
            SELECT MAX(timestamp) AS as_of
            FROM dim_time
            """
        ).fetchone()

        if (
            warehouse_as_of is None
            or warehouse_as_of["as_of"] is None
        ):
            raise HTTPException(
                status_code=500,
                detail=(
                    "Analytics warehouse contains "
                    "no AS_OF timestamp"
                ),
            )

        as_of = str(
            warehouse_as_of["as_of"]
        )

        # ----------------------------------------------------
        # Analytics freshness
        #
        # Measures the age of the analytics layer relative to
        # the current API request time.
        # ----------------------------------------------------
        try:
            as_of_dt = datetime.fromisoformat(
                as_of
            )

            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(
                    tzinfo=timezone.utc
                )

            now_dt = datetime.now(
                timezone.utc
            )

            analytics_freshness_hours = (
                now_dt - as_of_dt.astimezone(
                    timezone.utc
                )
            ).total_seconds() / 3600

            analytics_freshness_hours = max(
                0.0,
                analytics_freshness_hours,
            )

        except (ValueError, TypeError):
            analytics_freshness_hours = 0.0

        return PipelineStatusResponse(
            run_id=str(
                status_data["run_id"]
            ),
            run_timestamp=str(
                status_data["run_timestamp"]
            ),
            status=overall_status,
            healthy=healthy,
            reasons=reasons,
            tasks=tasks,
            rows_in=int(
                status_data["rows_in"]
            ),
            rows_rejected=int(
                status_data["rows_rejected"]
            ),
            nulls_handled=(
                None
                if status_data.get(
                    "nulls_handled"
                ) is None
                else int(
                    status_data["nulls_handled"]
                )
            ),
            rows_published=int(
                status_data["rows_published"]
            ),
            as_of=as_of,
            analytics_freshness_hours=round(
                analytics_freshness_hours,
                2,
            ),
            dim_time_rows=int(
                status_data["dim_time_rows"]
            ),
            dim_grid_rows=int(
                status_data["dim_grid_rows"]
            ),
            fact_rows=int(
                status_data["fact_rows"]
            ),
            orphan_time_keys=int(
                status_data["orphan_time_keys"]
            ),
            orphan_grid_keys=int(
                status_data["orphan_grid_keys"]
            ),
        )

    finally:
        connection.close()
@app.get(
    "/network/grid/{grid_id}/location",
    response_model=GridLocationResponse,
)
def get_grid_location(grid_id: int):

    if grid_id < 1 or grid_id > 10000:
        raise HTTPException(
            status_code=404,
            detail=f"Grid {grid_id} not found",
        )

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """
            SELECT grid_id, geometry
            FROM dim_grid
            WHERE grid_id = ?
            """,
            (grid_id,),
        ).fetchone()

        conn.close()

    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Warehouse error: {exc}",
        )

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Grid {grid_id} not found",
        )

    geometry = row["geometry"]

    if not geometry:
        raise HTTPException(
            status_code=500,
            detail=f"Geometry unavailable for grid {grid_id}",
        )

    try:
        geometry_data = json.loads(geometry)
    except json.JSONDecodeError:
        try:
            geometry_data = ast.literal_eval(geometry)
        except (ValueError, SyntaxError) as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Invalid geometry for grid {grid_id}: "
                    f"{exc}"
                ),
            )

    coordinates = geometry_data.get("coordinates")

    if not coordinates:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Geometry coordinates unavailable "
                f"for grid {grid_id}"
            ),
        )

    points = []

    def collect_points(value):
        if (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            points.append((float(value[0]), float(value[1])))
            return

        if isinstance(value, (list, tuple)):
            for item in value:
                collect_points(item)

    collect_points(coordinates)

    if not points:
        raise HTTPException(
            status_code=500,
            detail=f"No coordinate points found for grid {grid_id}",
        )

    longitudes = [point[0] for point in points]
    latitudes = [point[1] for point in points]

    centroid_longitude = (
        min(longitudes) + max(longitudes)
    ) / 2

    centroid_latitude = (
        min(latitudes) + max(latitudes)
    ) / 2

    return GridLocationResponse(
        grid_id=grid_id,
        polygon_reference=f"dim_grid.geometry:grid_{grid_id}",
        centroid_latitude=centroid_latitude,
        centroid_longitude=centroid_longitude,
    )