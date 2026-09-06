PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS fact_network_activity;
DROP TABLE IF EXISTS dim_grid;
DROP TABLE IF EXISTS dim_time;

CREATE TABLE dim_time (
    time_id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL UNIQUE,
    date TEXT NOT NULL,
    hour INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL
);

CREATE TABLE dim_grid (
    grid_id INTEGER PRIMARY KEY,
    geometry TEXT
);

CREATE TABLE fact_network_activity (
    time_id INTEGER NOT NULL,
    grid_id INTEGER NOT NULL,
    sms_in REAL NOT NULL,
    sms_out REAL NOT NULL,
    call_in REAL NOT NULL,
    call_out REAL NOT NULL,
    internet_activity REAL NOT NULL,
    total_activity REAL NOT NULL,

    PRIMARY KEY (time_id, grid_id),

    FOREIGN KEY (time_id) REFERENCES dim_time(time_id),
    FOREIGN KEY (grid_id) REFERENCES dim_grid(grid_id)
);

CREATE INDEX idx_fact_grid
ON fact_network_activity(grid_id);

CREATE INDEX idx_fact_time
ON fact_network_activity(time_id);

CREATE INDEX idx_fact_grid_time
ON fact_network_activity(grid_id, time_id);

CREATE INDEX idx_dim_time_date
ON dim_time(date);

CREATE INDEX idx_dim_time_hour
ON dim_time(hour);
