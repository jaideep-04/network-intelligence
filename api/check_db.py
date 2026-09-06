import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "warehouse" / "network_analytics.db"

print("=" * 70)
print("DATABASE SCHEMA")
print("=" * 70)
print(f"Database: {DB}")
print()

conn = sqlite3.connect(DB)

tables = conn.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name
""").fetchall()

for (table_name,) in tables:
    print(f"\nTABLE: {table_name}")
    print("-" * 70)

    columns = conn.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    for column in columns:
        print(
            f"  {column[1]:25} "
            f"{column[2]:15} "
            f"NOT NULL={bool(column[3])} "
            f"PK={bool(column[5])}"
        )

conn.close()

print("\n" + "=" * 70)
print("SCHEMA CHECK COMPLETED")
print("=" * 70)