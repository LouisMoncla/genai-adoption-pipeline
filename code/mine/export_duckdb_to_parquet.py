"""
Export x28_dump.duckdb (14 normalized relational tables) into Parquet files
matching the nested schema pipeline/data_preparation.py expects (per
schema_ads.json and config.yaml's column_mapping) — so the pipeline's existing
flattening code runs unmodified against this dataset.

Semantics established via inspection of the actual data (not assumed):
  - language: x28's own detected-language flag on `advertisements` (added
    2026-08-02, per Jeremias, to replace running FastText over the ad text
    downstream). Confirmed present on both source dumps via DESCRIBE. Values
    are lowercase ISO codes (de/fr/en/it/...) or NULL - NULL and any code not
    in config.yaml's allowed_languages (including near-zero "rm" coverage -
    x28 essentially never tags Romansh) fall out naturally via
    language_detection.py's existing allowed-language filter, no special
    handling needed.
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
    columns on `advertisements` (no join needed). company.metadata (industry
    tags) comes from company_metadata, filtered to type=='INDUSTRY' at export
    time (2026-08-02, see AGG_TABLE_SQL's agg_meta comment - the pipeline's
    own _flatten_and_filter() only ever reads INDUSTRY entries anyway, MARKET
    is dead weight downstream and was the main cost of an OOM on the largest
    source dump).
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

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb

# Parameterized via env vars (added 2026-07-23, to reuse this same tested
# script for a second source file - Jeremias shared a Dec 2020-Nov 2022
# dump to extend the pre-existing Dec 2022-Nov 2025 export). Defaults below
# exactly match the original hardcoded values, so a plain
# `python export_duckdb_to_parquet.py` run behaves identically to before -
# nothing about the tested per-day/per-month logic below changed, only how
# these constants get their values. OUT_DIR is deliberately NOT
# parameterized - both source files should land in the same output
# directory so Phase I sees one continuous set of day-files. SCRATCH_DB
# name IS parameterized so a second source doesn't reuse (and contaminate)
# the first one's agg_*/prejoined tables, which are specific to whichever
# database is ATTACHed as "src".
DB_PATH = os.environ.get(
    "EXPORT_DB_PATH", "C:/Users/Louis/Documents/Internship/data/raw/x28_dump.duckdb"
)
OUT_DIR = Path("C:/Users/Louis/Documents/Internship/data/processed/x28_parquet")
SCRATCH_DB = str(OUT_DIR / os.environ.get("EXPORT_SCRATCH_DB_NAME", "_scratch.duckdb"))

START_DATE = date.fromisoformat(os.environ.get("EXPORT_START_DATE", "2022-12-01"))
END_DATE = date.fromisoformat(os.environ.get("EXPORT_END_DATE", "2025-11-30"))


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


# Each table is its own statement (executed separately - see build_agg_tables)
# rather than one combined multi-statement string. A second source database
# (2026-07-23) OOM'd here even at a 1GB cap despite having FEWER rows in
# every underlying table than the original file that built these same four
# tables fine at 600MB - splitting them into separate execute() calls so
# each fully finishes (and its working memory is freed) before the next
# starts, rather than risking overlapping memory pressure from planning/
# running them as one batch.
AGG_TABLE_SQL = {
    "agg_occ": """
        CREATE OR REPLACE TABLE agg_occ AS
            SELECT advertisement_id,
                   list(struct_pack(id := metadata_id, name := name)) AS occupations
            FROM src.advertisement_metadata
            WHERE type = 'JOB' AND source = 'TITLE' AND name IS NOT NULL
            GROUP BY advertisement_id;
    """,
    # Split singleton vs. duplicated advertisement_ids (2026-08-02): on the
    # main 2022-2025 dump, advertisement_country has 6,575,794 ids with
    # exactly 1 row and only 2,565 with >1 (checked directly). A plain
    # `list(struct_pack(...)) ... GROUP BY` over the whole table OOM'd at
    # every memory cap tried (up to 2.5GB, more than agg_occ needed despite
    # agg_occ's filtered source having MORE rows) - the cost is per-group
    # LIST-aggregate-state overhead across 6.58M groups, not real data volume
    # (each list is 1 tiny struct). Building the singleton branch as a plain
    # per-row scalar list literal (no aggregation at all) and reserving the
    # real GROUP BY + list() aggregate for only the ~2,565 actually-duplicated
    # ids keeps the expensive path tiny. agg_cantons/agg_meta don't get the
    # same treatment - checked their group-size distributions and both have
    # real, wide multi-row fan-out (up to 27 cantons/ad, up to 621K ids with
    # 3+ metadata rows), so this singleton trick wouldn't help there anyway.
    "agg_loc": """
        CREATE OR REPLACE TABLE agg_loc AS
        WITH dup_ids AS (
            SELECT advertisement_id FROM src.advertisement_country
            GROUP BY advertisement_id HAVING COUNT(*) > 1
        )
        SELECT advertisement_id,
               list(struct_pack(
                   country := country,
                   province := CAST(NULL AS VARCHAR),
                   district := CAST(NULL AS INTEGER),
                   postal_code := CAST(NULL AS VARCHAR)
               )) AS locations
        FROM src.advertisement_country
        WHERE advertisement_id IN (SELECT advertisement_id FROM dup_ids)
        GROUP BY advertisement_id
        UNION ALL
        SELECT advertisement_id,
               [struct_pack(
                   country := country,
                   province := CAST(NULL AS VARCHAR),
                   district := CAST(NULL AS INTEGER),
                   postal_code := CAST(NULL AS VARCHAR)
               )] AS locations
        FROM src.advertisement_country
        WHERE advertisement_id NOT IN (SELECT advertisement_id FROM dup_ids);
    """,
    "agg_cantons": """
        CREATE OR REPLACE TABLE agg_cantons AS
            SELECT advertisement_id, list(DISTINCT canton) AS cantons_bonus
            FROM src.advertisement_canton
            GROUP BY advertisement_id;
    """,
    # Filtered to type='INDUSTRY' (2026-08-02): company_metadata has two types,
    # INDUSTRY (7.55M rows) and MARKET (6.28M) - grepped the whole pipeline/
    # and code/mine/ and confirmed data_preparation.py's _flatten_and_filter()
    # is the ONLY reader of company.metadata anywhere downstream, and it only
    # ever extracts the first type=="INDUSTRY" entry (MARKET is never read by
    # anything). Dropping MARKET at export time - not scope creep, it's
    # already-dead data on the main dump - roughly halves this table before
    # aggregation. Combined with the same singleton/duplicate split used for
    # agg_loc (INDUSTRY-only: 5,886,726 of 6,648,029 ids are singletons,
    # 88.5%), this was needed because the plain unfiltered
    # `list(struct_pack(...)) ... GROUP BY` OOM'd at every cap tried up to
    # 2.5GB - more than this machine has free to give it.
    "agg_meta": """
        CREATE OR REPLACE TABLE agg_meta AS
        WITH industry AS (
            SELECT advertisement_id, metadata_id, name, type
            FROM src.company_metadata WHERE type = 'INDUSTRY'
        ),
        dup_ids AS (
            SELECT advertisement_id FROM industry
            GROUP BY advertisement_id HAVING COUNT(*) > 1
        )
        SELECT advertisement_id,
               list(struct_pack(id := metadata_id, name := name, type := type)) AS company_metadata_list
        FROM industry
        WHERE advertisement_id IN (SELECT advertisement_id FROM dup_ids)
        GROUP BY advertisement_id
        UNION ALL
        SELECT advertisement_id,
               [struct_pack(id := metadata_id, name := name, type := type)] AS company_metadata_list
        FROM industry
        WHERE advertisement_id NOT IN (SELECT advertisement_id FROM dup_ids);
    """,
}

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
    a.language AS language,
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
         origin, company, occupations, language
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
    con.execute(f"SET memory_limit='{os.environ.get('EXPORT_MEMORY_LIMIT', '1GB')}'")
    temp_dir = OUT_DIR / "_duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{str(temp_dir).replace(chr(92), '/')}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=1")
    return con


def build_agg_tables(con):
    """Takes an already-open connection — does NOT open its own on entry. An
    earlier version opened a separate connection just for this check, then
    closed it and opened another for the day loop: two full connect+attach
    cycles (each slow under this machine's memory pressure) before any real
    query ran. That doubled overhead lines up with runs consistently dying
    right as the first day query started. Reusing one connection on entry
    removes that.

    Returns the (possibly reconnected) connection — see the reconnect-between-
    tables note below. Caller must use the returned value, same convention as
    build_prejoined_table().

    2026-08-02: reconnects between EACH table now, not just once at the end.
    Found the hard way re-running this on the largest (main 2022-2025) source
    dump for the first time since Step A got this dump's own dedicated
    per-loop reconnect fixes (build_prejoined_table's MONTHS_PER_CONNECTION,
    export_one_day's DAYS_PER_CONNECTION) — agg_loc OOM'd building a plain
    ~6.6M-row groupby (no join, no skew: 6,578,359 distinct keys from
    6,580,924 rows) right after agg_occ had just succeeded, at increasing
    memory_limit caps each retry (1.3/1.3, 1.8/1.8, 2.3/2.3 GiB - scaling with
    whatever cap was set, never with real headroom to spare). That's the exact
    "memory build-up across iterations within one connection" pattern already
    diagnosed and fixed for the day/month loops - Step A just never got the
    same fix, because on the two smaller source dumps tested earlier this
    never got stressed hard enough to surface it."""
    existing = {r[0] for r in con.sql("SHOW TABLES").fetchall()}
    for table_name, sql in AGG_TABLE_SQL.items():
        if table_name in existing:
            print(f"Step A: {table_name} already exists — skipping.", flush=True)
            continue
        print(f"Step A: building {table_name}...", flush=True)
        con.execute(sql)
        print(f"  {table_name}: done.", flush=True)
        con.close()
        con = _connect()
    return con


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
    con = build_agg_tables(con)
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
        con = build_agg_tables(con)
        con = build_prejoined_table(con)
        n = export_one_day(con, date(year, month, day_num))
        con.close()
        print(f"{year}-{month:02d}-{day_num:02d}: {n:,} rows", flush=True)
        return

    run_all_days()


if __name__ == "__main__":
    main()
