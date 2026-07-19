import duckdb

con = duckdb.connect('C:/Users/Louis/Documents/Internship/data/raw/x28_dump.duckdb', read_only=True)

for t in ['advertisement_positions', 'advertisement_education_levels', 'company_addresses']:
    print(f'=== {t} ===')
    for row in con.sql(f'DESCRIBE {t}').fetchall():
        print(row)
    print()

print('=== distinct types in advertisement_metadata ===')
print(con.sql("SELECT type, COUNT(*) FROM advertisement_metadata GROUP BY type").fetchall())

print()
print('=== distinct types in company_metadata ===')
print(con.sql("SELECT type, COUNT(*) FROM company_metadata GROUP BY type").fetchall())

print()
print('=== sample row from advertisement_details ===')
print(con.sql("SELECT * FROM advertisement_details LIMIT 1").fetchall())

print()
print('=== date range in advertisements.created ===')
print(con.sql("SELECT MIN(created), MAX(created) FROM advertisements").fetchall())
