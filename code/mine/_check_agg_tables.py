import duckdb

con = duckdb.connect('C:/Users/Louis/Documents/Internship/data/raw/x28_dump.duckdb', read_only=True)
tables = [r[0] for r in con.sql("SHOW TABLES").fetchall()]
print("agg_occ present:", "agg_occ" in tables)
print("agg_loc present:", "agg_loc" in tables)
print("agg_cantons present:", "agg_cantons" in tables)
print("agg_meta present:", "agg_meta" in tables)
if "agg_occ" in tables:
    print("agg_occ row count:", con.sql("SELECT COUNT(*) FROM agg_occ").fetchone()[0])
