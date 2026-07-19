import duckdb

con = duckdb.connect('C:/Users/Louis/Documents/Internship/data/raw/x28_dump.duckdb', read_only=True)

print('=== advertisement_positions sample ===')
print(con.sql("SELECT * FROM advertisement_positions LIMIT 10").fetchall())
print('distinct position count:', con.sql("SELECT COUNT(DISTINCT position) FROM advertisement_positions").fetchall())
print('positions per ad (max, avg):', con.sql("""
    SELECT MAX(c), AVG(c) FROM (
        SELECT advertisement_id, COUNT(*) c FROM advertisement_positions GROUP BY advertisement_id
    )
""").fetchall())

print()
print('=== advertisement_metadata type=JOB sample ===')
print(con.sql("SELECT * FROM advertisement_metadata WHERE type='JOB' LIMIT 10").fetchall())

print()
print('=== company_recruitment_agency distinct values ===')
print(con.sql("SELECT company_recruitment_agency, COUNT(*) FROM advertisements GROUP BY company_recruitment_agency").fetchall())

print()
print('=== advertisement_canton sample + cardinality ===')
print(con.sql("SELECT * FROM advertisement_canton LIMIT 5").fetchall())
print('cantons per ad (max):', con.sql("""
    SELECT MAX(c) FROM (SELECT advertisement_id, COUNT(*) c FROM advertisement_canton GROUP BY advertisement_id)
""").fetchall())

print()
print('=== advertisement_country cardinality ===')
print('countries per ad (max):', con.sql("""
    SELECT MAX(c) FROM (SELECT advertisement_id, COUNT(*) c FROM advertisement_country GROUP BY advertisement_id)
""").fetchall())

print()
print('=== advertisement_postalcode cardinality ===')
print('postalcodes per ad (max):', con.sql("""
    SELECT MAX(c) FROM (SELECT advertisement_id, COUNT(*) c FROM advertisement_postalcode GROUP BY advertisement_id)
""").fetchall())

print()
print('=== company_metadata INDUSTRY sample ===')
print(con.sql("SELECT * FROM company_metadata WHERE type='INDUSTRY' LIMIT 5").fetchall())

print()
print('=== how many ads have NULL raw_text or NULL created ===')
print(con.sql("""
    SELECT
      SUM(CASE WHEN d.raw_text IS NULL THEN 1 ELSE 0 END) AS null_text,
      SUM(CASE WHEN a.created IS NULL THEN 1 ELSE 0 END) AS null_created,
      COUNT(*) AS total
    FROM advertisements a
    LEFT JOIN advertisement_details d ON a.id = d.advertisement_id
""").fetchall())

print()
print('=== does every advertisement have exactly one advertisement_details row? ===')
print(con.sql("""
    SELECT COUNT(*) FROM (
        SELECT advertisement_id, COUNT(*) c FROM advertisement_details GROUP BY advertisement_id HAVING c > 1
    )
""").fetchall())
