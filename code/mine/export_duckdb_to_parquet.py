"""
Export x28_dump.duckdb (14 normalized relational tables) into Parquet files
matching the nested schema pipeline/data_preparation.py expects (per
schema_ads.json and config.yaml's column_mapping) — so the pipeline's existing
flattening code runs unmodified against this dataset.

Semantics established via inspection of the actual data (not assumed):
  - Occupation title lives in advertisement_metadata where type='JOB' AND
    source='TITLE' (name field). advertisement_positions is NOT occupation —
    it only has 7 distinct values (EMPLOYEE, DIRECTOR, ...), i.e. seniority level.
  - locations is built from advertisement_country alone. The pipeline only ever
    reads locations[].country (for the country_filter check) — canton and
    postal code are NOT joined into this struct because they have wildly
    different one-to-many cardinalities per ad (canton up to 27, postalcode up
    to 476) and joining three independent one-to-many tables directly would
    create a cartesian-product blowup. Canton is exported separately as a
    bonus, non-schema column in case cantonal analysis is wanted later.
  - company.{id,name,size,is_recruiter} come directly from the already-flat
    columns on `advertisements` (no join needed). company.metadata (industry/
    market tags) comes from company_metadata, passed through un-filtered —
    the pipeline's own _flatten_and_filter() already picks out type=="INDUSTRY".
  - date columns are formatted as strings ("%Y-%m-%d %H:%M:%S.%f") because
    data_preparation.py's _flatten_and_filter() calls .str.replace(...).str.to_datetime(...)
    on the date column, which requires a String dtype, not a native Timestamp.

Memory design (this machine has only 7.7 GB total RAM, and as little as
~1.4 GB actually FREE once Chrome/Defender/normal background usage is
accounted for — checked directly with Get-CimInstance Win32_OperatingSystem):
  1. Materialize each one-to-many aggregation (occupations, locations,
     cantons, company metadata) as its own finished table first, via
     CREATE TABLE AS — so DuckDB fully computes and frees each one before
     starting the next, instead of holding several large hash tables in
     memory at once. (Fixes: one combined query with 4 CTEs + the join OOM'd
     immediately.)
  2. Chunk the actual export by DAY, not year or month. A full month
     (~150-200K rows) still OOM'd even at an 800MB cap, with usage climbing
     past 763MB before failing — genuinely more than fits in the memory this
     machine actually has free. A day (~6K rows on average) is roughly 30x
     smaller.
  3. Open a FRESH DuckDB connection for every single day, inside one
     long-running Python process (not a fresh OS process per day — process
     spawn overhead across ~1,090 days added up too much). Evidence for why
     a fresh connection matters: an earlier attempt reused one connection
     across many months and OOM'd on the batch immediately AFTER a
     similar-sized batch had just succeeded — memory was building up across
     iterations within that one connection, not that any single batch was
     inherently too big.
  4. A separate, harder-to-explain failure mode: several attempts were
     silently killed with no Python traceback at all — not a graceful
     DuckDB OOM, just gone. This happened consistently within about a
     minute, regardless of memory_limit (tried 3GB, 800MB — same result),
     and did NOT happen when a run was started in the foreground first. The
     working theory is some environment-level watchdog killing long
     silent background processes. Printing progress after every single day
     (flush=True) keeps stdout active throughout, which avoids the pattern
     observed so far.
  Output is written to data/processed/, not data/raw/, because this Parquet
  is a derived/transformed product, not the original source file (the
  original x28_dump.duckdb in data/raw/ is untouched throughout).

  Earlier version of this script briefly opened x28_dump.duckdb directly
  with read_only=False to create the agg_* tables, then got interrupted
  mid-run — leaving those tables written into the RAW file, which violates
  "never edit raw data." Verified afterward that Step A had fully completed
  and dropped the leftover tables, restoring the raw file to its original
  14-table state. Fixed properly here: this script connects to a scratch
  database file and ATTACHes x28_dump.duckdb as read-only ("src") — the
  agg_* tables live only in the scratch db, and there is no code path left
  that can write to the raw file.

Run with the pipeline's venv (needs duckdb, installed separately — see
requirements.txt note).
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb

DB_PATH = "C:/Users/Louis/Documents/Internship/data/raw/x28_dump.duckdb"
OUT_DIR = Path("C:/Users/Louis/Documents/Internship/data/processed/x28_parquet")
# A persistent scratch FILE, not ":memory:" — survives a crash so Step A
# (which scans tables up to 77M rows) doesn't need to be redone on every
# retry. Disposable derived working state, never the raw data.
SCRATCH_DB = str(OUT_DIR / "_scratch.duckdb")

# The x28 dataset covers Dec 2022 - Nov 2025 (confirmed via
# SELECT MIN(created), MAX(created) FROM advertisements earlier). Using a
# slightly wider window is harmless — days with zero matching rows just
# produce a 0-row Parquet file quickly.
START_DATE = date(2022, 12, 1)
END_DATE = date(2025, 11, 30)


def _daterange():
    d = START_DATE
    while d <= END_DATE:
        yield d
        d += timedelta(days=1)


# --- Step A: materialize each one-to-many aggregation as its own table -----
# Reads from src.<table> (the ATTACHed, read-only raw database) and writes
# into the scratch database — never into src itself.
BUILD_AGG_TABLES_SQL = """
CREATE OR REPLACE TABLE agg_occ AS
    SELECT advertisement_id,
           list(struct_pack(id := metadata_id, name := name)) AS occupations
    FROM src.advertisement_metadata
    WHERE type = 'JOB' AND source = 'TITLE' AND name IS NOT NULL
    GROUP BY advertisement_id;

CREATE OR REPLACE TABLE agg_loc AS
    SELECT advertisement_id,
           list(struct_pack(
               country := country,
               province := CAST(NULL AS VARCHAR),
               district := CAST(NULL AS INTEGER),
               postal_code := CAST(NULL AS VARCHAR)
           )) AS locations
    FROM src.advertisement_country
    GROUP BY advertisement_id;

CREATE OR REPLACE TABLE agg_cantons AS
    SELECT advertisement_id, list(DISTINCT canton) AS cantons_bonus
    FROM src.advertisement_canton
    GROUP BY advertisement_id;

CREATE OR REPLACE TABLE agg_meta AS
    SELECT advertisement_id,
           list(struct_pack(id := metadata_id, name := name, type := type)) AS company_metadata_list
    FROM src.company_metadata
    GROUP BY advertisement_id;
"""

# --- Step B: single-day join against the pre-built aggregate tables --------
EXPORT_DAY_SQL = """
COPY (
  SELECT
    a.duplicategroup AS duplicate_group,
    d.title AS title,
    a.url AS url,
    strftime(a.created, '%Y-%m-%d %H:%M:%S.%f') AS tst_created,
    strftime(a.deleted, '%Y-%m-%d %H:%M:%S.%f') AS tst_deleted,
    CAST(NULL AS DOUBLE) AS duration,
    a.temporary AS is_temporary,
    a.homeoffice AS has_homeoffice,
    struct_pack(minimum := a.workquota_minimum, maximum := a.workquota_maximum) AS work_quota,
    CAST(NULL AS VARCHAR) AS location,
    COALESCE(loc.locations, []) AS locations,
    COALESCE(cantons.cantons_bonus, []) AS cantons_bonus,
    d.raw_text AS content_clean,
    a.origin AS origin,
    struct_pack(
        id := a.company_id,
        name := a.company_name,
        bfs_uid := a.company_uid,
        crn := a.company_crn,
        metadata := COALESCE(meta.company_metadata_list, []),
        addresses := CAST([] AS STRUCT(country VARCHAR, city VARCHAR, postal_code VARCHAR, street VARCHAR, is_primary BOOLEAN)[]),
        size := struct_pack(
            id := a.company_size_id,
            name := a.company_size_name,
            employees := struct_pack(minimum := a.company_size_min, maximum := a.company_size_max)
        ),
        url := a.company_url,
        is_recruiter := a.company_recruitment_agency
    ) AS company,
    COALESCE(occ.occupations, []) AS occupations
  FROM src.advertisements a
  JOIN src.advertisement_details d ON a.id = d.advertisement_id
  LEFT JOIN agg_occ occ ON a.id = occ.advertisement_id
  LEFT JOIN agg_loc loc ON a.id = loc.advertisement_id
  LEFT JOIN agg_cantons cantons ON a.id = cantons.advertisement_id
  LEFT JOIN agg_meta meta ON a.id = meta.advertisement_id
  WHERE CAST(a.created AS DATE) = DATE '{day}'
) TO '{out_file}' (FORMAT PARQUET)
"""


def _connect():
    # ATTACHing the 23GB raw file has taken up to ~2 minutes under this
    # machine's memory pressure, with zero output in between — printing
    # before/after keeps stdout active through that gap instead of going
    # silent for the whole window.
    print("  (opening connection + attaching source database...)", flush=True)
    con = duckdb.connect(SCRATCH_DB)
    con.execute(f"ATTACH '{DB_PATH}' AS src (READ_ONLY)")
    print("  (connected.)", flush=True)
    con.execute("SET memory_limit='600MB'")  # comfortably under the ~1.4GB actually free
    temp_dir = OUT_DIR / "_duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{str(temp_dir).replace(chr(92), '/')}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=1")
    return con


def build_agg_tables():
    """Step A, run once. Cached in the persistent scratch db across process runs."""
    con = _connect()
    existing = {r[0] for r in con.sql("SHOW TABLES").fetchall()}
    if {"agg_occ", "agg_loc", "agg_cantons", "agg_meta"}.issubset(existing):
        print("Step A: agg_* tables already exist — skipping.", flush=True)
    else:
        print("Step A: materializing occupation/location/canton/company-metadata aggregate tables...", flush=True)
        con.execute(BUILD_AGG_TABLES_SQL)
        print("  Done.", flush=True)
    con.close()


def _check_existing(out_file: Path) -> int | None:
    """Returns row count if out_file is a valid, already-complete export;
    None if it needs (re-)exporting (deletes it first if corrupt)."""
    if not out_file.exists():
        return None
    try:
        con = duckdb.connect(":memory:")
        n = con.sql(f"SELECT COUNT(*) FROM read_parquet('{str(out_file).replace(chr(92), '/')}')").fetchone()[0]
        con.close()
        return n
    except duckdb.Error:
        print(f"  {out_file.name}: existing file is corrupt/truncated — re-exporting.", flush=True)
        out_file.unlink()
        return None


def export_one_day(con, day: date) -> int:
    """Step B for a single day, using an already-open connection passed in by
    the caller (see run_all_days — connections are reused across a batch of
    days, not recreated per day, because opening the 23GB read-only attach
    is itself slow on this memory-pressured machine: ~2 minutes just to
    connect in one test, far more than the query itself needed)."""
    day_str = day.isoformat()
    out_file = OUT_DIR / f"x28_ads_{day_str}.parquet"

    cached = _check_existing(out_file)
    if cached is not None:
        return cached

    # Write to a temp filename, rename only after a successful COPY — a crash
    # mid-write must never leave a broken file at the real output name (this
    # happened once with the month-based version of this script).
    tmp_file = out_file.with_suffix(".parquet.inprogress")
    con.execute(EXPORT_DAY_SQL.format(day=day_str, out_file=str(tmp_file).replace("\\", "/")))
    n = con.sql(f"SELECT COUNT(*) FROM read_parquet('{str(tmp_file).replace(chr(92), '/')}')").fetchone()[0]
    tmp_file.replace(out_file)  # atomic on the same volume
    return n


# How many days to process per connection before closing and reopening it.
# Balances two failure modes seen during testing: (a) a fresh connection per
# day is safe against memory build-up but each connect+attach took ~2 minutes
# under this machine's memory pressure — far too slow across ~1,090 days;
# (b) one connection reused for a whole month-sized loop (48 iterations)
# OOM'd right after a same-sized batch had just succeeded. Day-sized batches
# are roughly 30x smaller than month-sized ones, so resetting every 30 days
# (instead of never) should land well on the safe side of that boundary
# while cutting connection overhead ~30x versus one-per-day.
DAYS_PER_CONNECTION = 30


def run_all_days():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_agg_tables()

    all_days = list(_daterange())
    total_rows = 0
    days_with_data = 0
    con = None

    for i, day in enumerate(all_days):
        # Skip the connection dance entirely for days already done — no need
        # to even open a connection just to check a cached result.
        out_file = OUT_DIR / f"x28_ads_{day.isoformat()}.parquet"
        cached = _check_existing(out_file)
        was_cached = cached is not None
        if was_cached:
            n = cached
        else:
            if con is None or i % DAYS_PER_CONNECTION == 0:
                if con is not None:
                    con.close()
                con = _connect()
            n = export_one_day(con, day)

        total_rows += n
        if n > 0:
            days_with_data += 1

        if (i + 1) % 20 == 0 or (not was_cached and n > 0):
            print(f"  [{i+1}/{len(all_days)}] {day.isoformat()}: {n:,} rows "
                  f"(running total {total_rows:,})", flush=True)

    if con is not None:
        con.close()

    print(f"Done. {days_with_data} days had data. Total exported rows: {total_rows:,}", flush=True)

    con = _connect()
    expected = con.sql("SELECT COUNT(*) FROM src.advertisements").fetchone()[0]
    con.close()
    print(f"Sanity check — advertisements table row count: {expected:,} "
          f"({'MATCH' if expected == total_rows else 'MISMATCH — investigate'})", flush=True)


def main():
    if len(sys.argv) == 4:
        # Single-day mode, for manual testing: python export_....py 2023 1 15
        year, month, day_num = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        con = _connect()
        n = export_one_day(con, date(year, month, day_num))
        con.close()
        print(f"{year}-{month:02d}-{day_num:02d}: {n:,} rows", flush=True)
        return

    run_all_days()


if __name__ == "__main__":
    main()
