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

Design history (this machine has only 7.7 GB total RAM, and as little as
~1.1-1.4 GB actually FREE once Chrome/Defender/normal background usage is
accounted for):
  1. Materialize each one-to-many aggregation (occupations, locations,
     cantons, company metadata) as its own finished table first, via
     CREATE TABLE AS — so DuckDB fully computes and frees each one before
     starting the next. Kept throughout every version of this script.
  2. Tried year, then month, then day chunking via WHERE date filters, then
     tried ONE single unpartitioned pass instead (theorizing that WHERE
     filters on an unordered table cost as much as a full scan anyway). The
     single pass got killed by something outside this script's control
     (not a DuckDB OOM — confirmed by real, growing disk-spill progress
     each time, 1.8-2.2GB, right up until the kill) after a few minutes,
     every attempt, losing 100% of progress every time since a single COPY
     is all-or-nothing.
  3. Landed here: per-day chunking (~6K rows/day on average — small and
     fast), because it's the only approach that produces genuinely DURABLE
     partial progress — each completed day is a real, permanently-valid
     Parquet file, so a kill only costs the one day in flight, not
     everything. A manual single-day test did complete successfully
     end-to-end, confirming day-sized queries finish in reasonable time
     once connected. Connections are reused across a batch of days (not
     recreated per day) because opening the 23GB read-only ATTACH is itself
     slow (~2 min the first time; much faster once the OS file cache is
     warm) — but reset periodically (see DAYS_PER_CONNECTION) because an
     earlier attempt reusing one connection for a whole month-sized loop
     OOM'd right after a similar-sized batch had just succeeded, suggesting
     memory build-up across iterations within one connection.
  Practically: this script is meant to be re-run repeatedly. Every rerun
  skips whatever days are already done (verified valid, not just present —
  see _check_existing) and only works on what's left. Progress accumulates
  across retries even if most individual runs get killed partway through.
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

  4. (2026-07-22) Profiled WHY per-day export was taking ~90s/day for as
     little as ~6K rows: a plain date-filtered COUNT on `advertisements`
     took ~0.1s, and adding the `advertisement_details` join only brought it
     to ~0.6s — so neither of those was the cost. That leaves the four LEFT
     JOINs against agg_occ/agg_loc/agg_cantons/agg_meta, which are
     multi-million-row tables that don't change but were being rebuilt into
     join structures from scratch on EVERY one of the 1,096 day queries.
     Disk-spill usage was ~0MB throughout, confirming this was never a
     memory problem — the 600MB cap was never actually binding, so freeing
     RAM would not have helped.
     Added Step B: do those 4 joins ONCE PER MONTH (36 times) instead of
     once per day (1,096 times), writing the joined-but-not-yet-day-filtered
     result into a new `prejoined` table. Each day's export (Step C) then
     becomes a cheap single-table filter + COPY, no more joins at export
     time. Chunked by month rather than done in one pass, specifically
     because the one-single-pass attempt in design point 2 above is exactly
     this same join and got killed by something external after a few
     minutes every time — a month-sized chunk is ~30x smaller and, per the
     per-day timing evidence, should reliably finish well inside that
     window. Each month's INSERT is one all-or-nothing statement (same
     durability property as the day-level COPY), so a rerun can safely
     check "does `prejoined` already have rows for this month?" to skip
     completed months. Already-exported days are untouched by this change —
     Step C still skips any day that already has a valid Parquet file.

Run with the pipeline's venv (needs duckdb, installed separately — see
requirements.txt note).
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb

DB_PATH = "C:/Users/Louis/Documents/Internship/data/raw/x28_dump.duckdb"
OUT_DIR = Path("C:/Users/Louis/Documents/Internship/data/processed/x28_parquet")
SCRATCH_DB = str(OUT_DIR / "_scratch.duckdb")

START_DATE = date(2022, 12, 1)
END_DATE = date(2025, 11, 30)


def _daterange():
    d = START_DATE
    while d <= END_DATE:
        yield d
        d += timedelta(days=1)


def _month_starts():
    y, m = START_DATE.year, START_DATE.month
    while (y, m) <= (END_DATE.year, END_DATE.month):
        yield date(y, m, 1)
        m += 1
        if m > 12:
            m = 1
            y += 1


def _next_month(d: date) -> date:
    return date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)


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

# Shared SELECT for building `prejoined` — same 4 LEFT JOINs as the old
# per-day query, but run once per month (see design history point 4) and
# keeping the raw `created` timestamp column (not just the stringified
# tst_created) so Step C can still filter by exact day afterward.
_PREJOINED_SELECT = """
  SELECT
    a.duplicategroup AS duplicate_group,
    d.title AS title,
    a.url AS url,
    a.created AS created,
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
"""

CREATE_PREJOINED_SCHEMA_SQL = f"CREATE TABLE prejoined AS {_PREJOINED_SELECT} WHERE FALSE"

INSERT_PREJOINED_CHUNK_SQL = (
    f"INSERT INTO prejoined {_PREJOINED_SELECT}"
    " WHERE a.created >= TIMESTAMP '{chunk_start}' AND a.created < TIMESTAMP '{chunk_end}'"
    # No ORDER BY here on purpose: sorting a whole month (with the large
    # content_clean text + nested struct/list columns) needs more memory
    # than the 600MB cap allows and OOM'd in testing. Row-group pruning on
    # Step C's per-day filter is a bit less precise without it (rows are in
    # month-chunk order, not exact date order), but chunking by month still
    # keeps each day's filter scan cheap since it only has to scan ~1
    # month's worth of rows instead of the whole table.
)

# Step C: now a plain filter over the already-joined `prejoined` table — no
# more joins at export time, which is what makes this step fast.
EXPORT_DAY_SQL = """
COPY (
  SELECT duplicate_group, title, url, tst_created, tst_deleted, duration, is_temporary,
         has_homeoffice, work_quota, location, locations, cantons_bonus, content_clean,
         origin, company, occupations
  FROM prejoined
  WHERE CAST(created AS DATE) = DATE '{day}'
) TO '{out_file}' (FORMAT PARQUET)
"""


def _connect():
    # Printing before/after keeps stdout active during the connect+attach —
    # every silent multi-minute gap observed so far has ended in an external
    # kill with no Python traceback; adding these prints back in (they were
    # accidentally dropped in an earlier rewrite) measurably correlates with
    # runs surviving longer.
    print("  (opening connection + attaching source database...)", flush=True)
    con = duckdb.connect(SCRATCH_DB)
    con.execute(f"ATTACH '{DB_PATH}' AS src (READ_ONLY)")
    print("  (connected.)", flush=True)
    # 600MB was set cautiously back when the OLD monolithic-query design was
    # OOMing (design history point 2). It turned out too thin for Step B's
    # monthly joins: a real run OOM'd at 572/572.2MB on 2025-07 — and doing
    # so on the very first month tried right after a fresh reconnect ruled
    # out connection-buildup as the cause (see MONTHS_PER_CONNECTION). Some
    # months are just marginally heavier than others. There's real headroom
    # (~1.68GB free out of 7.7GB total, checked live) so raised to 1GB,
    # still leaving a safety margin for the OS/other processes.
    con.execute("SET memory_limit='1GB'")
    temp_dir = OUT_DIR / "_duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{str(temp_dir).replace(chr(92), '/')}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=1")
    return con


def build_agg_tables(con):
    """Takes an already-open connection — does NOT open its own. An earlier
    version opened a separate connection just for this check, then closed it
    and opened another for the day loop: two full connect+attach cycles
    (each slow under this machine's memory pressure) before any real query
    ran. That doubled overhead lines up with runs consistently dying right
    as the first day query started. Reusing one connection removes that."""
    existing = {r[0] for r in con.sql("SHOW TABLES").fetchall()}
    if {"agg_occ", "agg_loc", "agg_cantons", "agg_meta"}.issubset(existing):
        print("Step A: agg_* tables already exist — skipping.", flush=True)
    else:
        print("Step A: materializing occupation/location/canton/company-metadata aggregate tables...", flush=True)
        con.execute(BUILD_AGG_TABLES_SQL)
        print("  Done.", flush=True)


MONTHS_PER_CONNECTION = 6


def build_prejoined_table(con):
    """Step B — see design history point 4. Does the 4 expensive LEFT JOINs
    once per month instead of once per day. Safe to interrupt: each month's
    INSERT is one all-or-nothing statement, so re-running just re-checks
    which months already have rows and skips them.

    Returns the (possibly reconnected) connection — see MONTHS_PER_CONNECTION
    below. A live run OOM'd 572/572.2MB into month 34/36 despite each
    individual month comfortably fitting in the same 600MB cap on its own
    (month 1 of testing took 122s with no memory issue). That matches the
    exact "memory build-up across iterations within one connection" pattern
    already diagnosed for the old per-day loop (see DAYS_PER_CONNECTION) —
    this loop just never got the same periodic-reconnect fix when Step B was
    added. Fixed here the same way: reconnect every few months so state
    doesn't accumulate across 36 INSERTs on one connection."""
    existing = {r[0] for r in con.sql("SHOW TABLES").fetchall()}
    if "prejoined" not in existing:
        con.execute(CREATE_PREJOINED_SCHEMA_SQL)
        print("  Created empty `prejoined` table.", flush=True)

    months_done_this_run = 0
    for chunk_start in _month_starts():
        chunk_end = _next_month(chunk_start)
        n_existing = con.sql(
            f"SELECT COUNT(*) FROM prejoined WHERE created >= TIMESTAMP '{chunk_start}' "
            f"AND created < TIMESTAMP '{chunk_end}'"
        ).fetchone()[0]
        if n_existing > 0:
            print(f"Step B: {chunk_start} already joined ({n_existing:,} rows) — skipping.", flush=True)
            continue

        if months_done_this_run > 0 and months_done_this_run % MONTHS_PER_CONNECTION == 0:
            con.close()
            con = _connect()

        print(f"Step B: joining {chunk_start} ...", flush=True)
        con.execute(INSERT_PREJOINED_CHUNK_SQL.format(chunk_start=chunk_start, chunk_end=chunk_end))
        months_done_this_run += 1
        n = con.sql(
            f"SELECT COUNT(*) FROM prejoined WHERE created >= TIMESTAMP '{chunk_start}' "
            f"AND created < TIMESTAMP '{chunk_end}'"
        ).fetchone()[0]
        print(f"  {chunk_start}: {n:,} rows joined.", flush=True)

    return con


def _check_existing(out_file: Path) -> int | None:
    """Row count if out_file is a valid, complete export; None if it needs
    (re-)exporting (deletes it first if corrupt/truncated)."""
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
    day_str = day.isoformat()
    out_file = OUT_DIR / f"x28_ads_{day_str}.parquet"
    tmp_file = out_file.with_suffix(".parquet.inprogress")
    # Write to a temp filename, rename only after a successful COPY — a crash
    # mid-write must never leave a broken file at the real output name.
    con.execute(EXPORT_DAY_SQL.format(day=day_str, out_file=str(tmp_file).replace("\\", "/")))
    n = con.sql(f"SELECT COUNT(*) FROM read_parquet('{str(tmp_file).replace(chr(92), '/')}')").fetchone()[0]
    tmp_file.replace(out_file)  # atomic on the same volume
    return n


DAYS_PER_CONNECTION = 30


def run_all_days():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_days = list(_daterange())
    total_rows = 0
    days_with_data = 0
    days_done_this_run = 0

    # One connection covers Step A AND the first DAYS_PER_CONNECTION days —
    # see build_agg_tables docstring for why this matters.
    con = _connect()
    build_agg_tables(con)
    con = build_prejoined_table(con)

    for i, day in enumerate(all_days):
        out_file = OUT_DIR / f"x28_ads_{day.isoformat()}.parquet"
        cached = _check_existing(out_file)
        if cached is not None:
            n = cached
        else:
            if days_done_this_run > 0 and days_done_this_run % DAYS_PER_CONNECTION == 0:
                con.close()
                con = _connect()
            n = export_one_day(con, day)
            days_done_this_run += 1
            print(f"  [{i+1}/{len(all_days)}] {day.isoformat()}: {n:,} rows", flush=True)

        total_rows += n
        if n > 0:
            days_with_data += 1

    con.close()

    print(f"Run finished. {days_with_data} days had data so far. "
          f"Total rows across all completed days: {total_rows:,}. "
          f"({days_done_this_run} days newly processed this run.)", flush=True)

    con = _connect()
    expected = con.sql("SELECT COUNT(*) FROM src.advertisements").fetchone()[0]
    con.close()
    print(f"Sanity check — advertisements table row count: {expected:,} "
          f"({'MATCH — all done!' if expected == total_rows else 'still incomplete, rerun to continue'})",
          flush=True)


def main():
    if len(sys.argv) == 4:
        # Manual single-day test: python export_....py 2023 1 15
        year, month, day_num = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        con = _connect()
        build_agg_tables(con)
        con = build_prejoined_table(con)
        n = export_one_day(con, date(year, month, day_num))
        con.close()
        print(f"{year}-{month:02d}-{day_num:02d}: {n:,} rows", flush=True)
        return

    run_all_days()


if __name__ == "__main__":
    main()
