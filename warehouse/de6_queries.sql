-- ============================================================
-- DE6 - NETWORK ANALYTICS QUERIES
-- ============================================================

-- 1. TOP 10 GRIDS BY TOTAL NETWORK ACTIVITY
SELECT
    f.grid_id,
    ROUND(SUM(f.total_activity), 2) AS total_activity
FROM fact_network_activity f
GROUP BY f.grid_id
ORDER BY total_activity DESC
LIMIT 10;


-- 2. HOURLY ACTIVITY TREND
SELECT
    t.hour,
    ROUND(SUM(f.total_activity), 2) AS total_activity,
    ROUND(SUM(f.internet_activity), 2) AS internet_activity
FROM fact_network_activity f
JOIN dim_time t
    ON f.time_id = t.time_id
GROUP BY t.hour
ORDER BY t.hour;


-- 3. INTERNET-HEAVY WINDOWS
SELECT
    t.timestamp,
    f.grid_id,
    ROUND(f.internet_activity, 2) AS internet_activity,
    ROUND(f.total_activity, 2) AS total_activity
FROM fact_network_activity f
JOIN dim_time t
    ON f.time_id = t.time_id
ORDER BY f.internet_activity DESC
LIMIT 10;


-- 4. TOP GRIDS WITH INTERNET ACTIVITY
SELECT
    f.grid_id,
    ROUND(SUM(f.internet_activity), 2) AS total_internet_activity
FROM fact_network_activity f
GROUP BY f.grid_id
ORDER BY total_internet_activity DESC
LIMIT 10;


-- 5. VALIDATE FACT ↔ DIMENSION JOINS
SELECT
    COUNT(*) AS joined_rows
FROM fact_network_activity f
JOIN dim_time t
    ON f.time_id = t.time_id
JOIN dim_grid g
    ON f.grid_id = g.grid_id;
