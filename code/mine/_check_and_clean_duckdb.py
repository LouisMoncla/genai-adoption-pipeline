import duckdb

con = duckdb.connect('C:/Users/Louis/Documents/Internship/data/raw/x28_dump.duckdb', read_only=False)

print("=== Current tables ===")
tables = con.sql("SHOW TABLES").fetchall()
print(tables)

agg_tables = [t[0] for t in tables if t[0].startswith('agg_')]
if agg_tables:
    print(f"\nFound {len(agg_tables)} leftover agg_ tables from the killed run — dropping them: {agg_tables}")
    for t in agg_tables:
        con.execute(f"DROP TABLE IF EXISTS {t}")
    print("Dropped.")
else:
    print("\nNo agg_ tables found — nothing to clean up.")

print("\n=== Tables after cleanup ===")
print(con.sql("SHOW TABLES").fetchall())

con.close()

# Force a checkpoint/vacuum-like close isn't directly exposed; just confirm
# table list matches the original 14 tables.
