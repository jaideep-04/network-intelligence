import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "warehouse" / "network_analytics.db"

conn = sqlite3.connect(DB)

print("=" * 70)
print("PHASE 4 - API1 GROUND TRUTH")
print("=" * 70)

# 1. Latest timestamp
as_of = conn.execute("""
    SELECT MAX(timestamp)
    FROM dim_time
""").fetchone()[0]

print(f"\nAS_OF: {as_of}")

# 2. Total activity at AS_OF
total_activity = conn.execute("""
    SELECT SUM(f.total_activity)
    FROM fact_network_activity f
    JOIN dim_time t ON f.time_id = t.time_id
    WHERE t.timestamp = ?
""", (as_of,)).fetchone()[0]

print(f"TOTAL ACTIVITY: {total_activity}")

# 3. Active grids at AS_OF
active_grids = conn.execute("""
    SELECT COUNT(DISTINCT f.grid_id)
    FROM fact_network_activity f
    JOIN dim_time t ON f.time_id = t.time_id
    WHERE t.timestamp = ?
      AND f.total_activity > 0
""", (as_of,)).fetchone()[0]

print(f"ACTIVE GRIDS: {active_grids}")

# 4. Peak hour
peak = conn.execute("""
    SELECT
        t.timestamp,
        t.hour,
        SUM(f.total_activity) AS activity
    FROM fact_network_activity f
    JOIN dim_time t ON f.time_id = t.time_id
    GROUP BY t.timestamp, t.hour
    ORDER BY activity DESC
    LIMIT 1
""").fetchone()

print(f"PEAK HOUR TIMESTAMP: {peak[0]}")
print(f"PEAK HOUR: {peak[1]}")
print(f"PEAK ACTIVITY: {peak[2]}")

# 5. Top grid at AS_OF
top_grid = conn.execute("""
    SELECT
        f.grid_id,
        f.total_activity
    FROM fact_network_activity f
    JOIN dim_time t ON f.time_id = t.time_id
    WHERE t.timestamp = ?
    ORDER BY f.total_activity DESC, f.grid_id ASC
    LIMIT 1
""", (as_of,)).fetchone()

print(f"TOP GRID: {top_grid[0]}")
print(f"TOP GRID ACTIVITY: {top_grid[1]}")

conn.close()

print("\n" + "=" * 70)
print("GROUND TRUTH CHECK COMPLETED")
print("=" * 70)