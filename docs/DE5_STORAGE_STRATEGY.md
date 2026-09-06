# DE5 — Storage Strategy & Data Zones

## 1. Objective

Define a practical storage contract for the Network Intelligence pipeline covering
landing, raw, rejected, reference, processed, analytics, and logging zones.

The strategy is based on auditability, processing efficiency, query performance,
retention, operational simplicity, and clear write semantics.

---

## 2. Storage Strategy

| Zone | Purpose | Format | Write Mode | Partitioning | Retention |
|---|---|---|---|---|---|
| data/landing/ | Incoming daily files | CSV | Append / new files | None | Short-term |
| data/raw/ | Accepted source data | CSV | Immutable / append-only | None | Long-term |
| data/rejected/ | Invalid input files | CSV | Append | None | Short/medium-term |
| data/reference/ | Static Milan grid reference | GeoJSON | Read-only | None | Long-term |
| data/processed/ | Cleaned and transformed datasets | Parquet | Append by processing/date | Date-partitioned where applicable | Long-term |
| data/analytics/ | Curated analytical outputs | Parquet / warehouse tables | Controlled append or overwrite | Selective | Business-defined |
| logs/ | Audit and execution history | CSV / JSON / text | Append-only | Optional date-based organization | Medium/long-term |

---

## 3. Zone Definitions

### 3.1 Landing Zone

Location:

    data/landing/

Contains incoming daily telecom CSV files before validation.

Example:

    sms-call-internet-mi-2013-11-01.csv

Characteristics:

- Operational input zone.
- Files may be newly added each day.
- Files are validated before entering the raw zone.
- The static Milan GeoJSON must NOT enter this zone.

Write semantics:

    Append / new-file arrival

Retention:

    Short-term operational retention.
    Files may be removed after successful routing and verification,
    subject to operational/audit requirements.

---

### 3.2 Raw Zone

Location:

    data/raw/

Contains accepted source files.

Characteristics:

- Raw files are preserved unchanged.
- Raw data provides reproducibility and auditability.
- Downstream processing must read from this trusted raw layer.
- Raw files should never be modified in place.

Write semantics:

    Append-only / immutable

Retention:

    Long-term where storage permits.

Why raw must remain immutable:

If the original source is overwritten, it becomes difficult or impossible to:

- reproduce a historical pipeline run,
- investigate a data-quality issue,
- compare source data against transformed data,
- prove what data was originally received,
- replay processing using the exact original input.

---

### 3.3 Rejected Zone

Location:

    data/rejected/

Contains files that fail ingestion validation.

Characteristics:

- Invalid files are quarantined instead of entering raw.
- Rejection reason is recorded in the ingestion audit log.
- Rejected files can be inspected and corrected separately.

Write semantics:

    Append

Retention:

    Short/medium-term depending on audit requirements.

---

### 3.4 Reference Zone

Location:

    data/reference/

Primary asset:

    milano-grid.geojson

Characteristics:

- Static reference data.
- Used for geographic enrichment.
- Managed separately from daily telecom ingestion.
- Must not be treated as an operational daily input.

Write semantics:

    Read-only after controlled update.

Partitioning:

    NONE

The GeoJSON is static reference data and therefore should NOT be date-partitioned.

Retention:

    Long-term.

---

### 3.5 Processed Zone

Location:

    data/processed/

Contains cleaned and transformed Parquet datasets produced by Spark.

Examples:

    data/processed/activity/
    data/processed/sp2_clean/
    data/processed/sp3_hourly_grid/
    data/processed/sp4_geo_enriched/
    data/processed/sp5_performance/

Recommended date-partitioned structure:

    data/processed/
    └── activity/
        ├── date=2013-11-01/
        ├── date=2013-11-02/
        ├── date=2013-11-03/
        ├── date=2013-11-04/
        ├── date=2013-11-05/
        ├── date=2013-11-06/
        └── date=2013-11-07/

Write semantics:

    Append for new dates.
    Controlled overwrite only when intentionally rebuilding
    a specific partition.

Format:

    Parquet

Benefits:

- Columnar storage.
- Efficient Spark processing.
- Predicate filtering.
- Reduced storage footprint compared with CSV.
- Better schema preservation.

Important distinction:

Spark's part-00000, part-00001, etc. files are execution/output partitions.
They are NOT the same as business/date partitions.

---

### 3.6 Analytics Zone

Location:

    data/analytics/

Current analytical outputs include:

    data/analytics/hourly_grid_summary/
    data/analytics/grid_performance/
    data/analytics/temporal_performance/

Format:

    Parquet and/or warehouse tables

Purpose:

- Fast analytical access.
- Dashboard consumption.
- Network activity analysis.
- Downstream data products.

Write semantics:

    Controlled overwrite for deterministic derived datasets.
    Append when the dataset represents incrementally accumulated history.

Partitioning:

    Apply only where it provides query or operational value.
    Do not introduce unnecessary partitioning.

Retention:

    Business-defined based on reporting and analytical requirements.

---

### 3.7 Logs Zone

Location:

    logs/

Current files include:

    logs/ingestion_log.csv
    logs/spark_pipeline_log.csv
    logs/validation_results.json

Purpose:

- Ingestion audit history.
- Spark execution history.
- Validation results.
- Operational troubleshooting.

Write semantics:

    Append-only

Retention:

    Medium/long-term according to audit and operational requirements.

Logs must be treated as a first-class storage layer rather than merely
temporary debugging output.

---

## 4. Append vs Overwrite Contract

### Landing

Use:

    APPEND / NEW FILE

New files arrive without replacing previously received files.

### Raw

Use:

    IMMUTABLE / APPEND-ONLY

Existing accepted source files must not be overwritten.

### Rejected

Use:

    APPEND

Each rejected input should remain traceable.

### Reference

Use:

    CONTROLLED REPLACEMENT

Reference data is updated only through an explicit controlled update,
not through daily ingestion.

### Processed

Use:

    APPEND BY DATE PARTITION

For a new date, create a new partition.

For an intentional rebuild of one date, that specific partition may be
overwritten.

### Analytics

Use:

    CONTROLLED APPEND OR OVERWRITE

The choice depends on whether the analytical dataset is cumulative
or deterministically regenerated.

### Logs

Use:

    APPEND-ONLY

Existing audit history should not be silently overwritten.

---

## 5. Storage Contract

The project follows these rules:

1. Landing contains incoming operational files.
2. Raw contains accepted source files and preserves them unchanged.
3. Rejected contains invalid source files and their failure reasons are logged.
4. Reference contains static assets such as milano-grid.geojson.
5. Reference data is NOT date-partitioned.
6. Processed data is stored as Parquet.
7. Processed datasets may use date partitions where partitioning provides value.
8. Analytics contains curated outputs for downstream querying and applications.
9. Logs contain audit and execution history.
10. Raw data is immutable.
11. Write semantics are defined independently for each zone.
12. Spark execution partitions such as part-00000 are not business/date partitions.
13. Storage decisions prioritize auditability, performance, cost, and operational simplicity.

---

## 6. Current Project Storage

Current structure:

    data/
    ├── landing/
    ├── raw/
    ├── rejected/
    ├── reference/
    │   └── milano-grid.geojson
    ├── processed/
    │   ├── activity/
    │   ├── sp2_clean/
    │   ├── sp3_hourly_grid/
    │   ├── sp4_geo_enriched/
    │   └── sp5_performance/
    └── analytics/
        ├── hourly_grid_summary/
        ├── grid_performance/
        └── temporal_performance/

    logs/
    ├── ingestion_log.csv
    ├── spark_pipeline_log.csv
    └── validation_results.json

---

## 7. DE5 Acceptance Checklist

- [x] Every zone has a stated format.
- [x] Every zone has a stated write mode.
- [x] Every zone has a stated retention position.
- [x] Reference data is explicitly NOT date-partitioned.
- [x] logs/ is explicitly included in the storage strategy.
- [x] Raw data immutability is documented.
- [x] Parquet is documented for processed data.
- [x] Append vs overwrite semantics are defined per layer.
- [x] Date-partitioned processed directory design is documented.
- [x] Spark execution partitions are distinguished from business/date partitions.

## DE5 Status

COMPLETED
