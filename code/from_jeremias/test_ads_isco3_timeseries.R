# x28 vacancies: monthly ad counts and distinct ISCO-3 occupations
#
# Two monthly series:
#   (1) vacancies created per month (deduplicated)
#   (2) number of distinct ISCO-08 3-digit occupations per month
#
# Cleaning:
#   - drop apprenticeships
#   - deduplicate repostings: x28 tags repostings of the same vacancy with a
#     shared `duplicategroup`; keep the EARLIEST ad per group (ads with no
#     group are kept as-is)
#   - occupation: job id -> AVAM (x28_Occupation_to_AVAM.xlsx)
#                        -> ISCO-4 (q93_isco.csv) -> ISCO-3 = floor(ISCO-4 / 10)
#     ads with more than MAX_JOBS job ids are dropped as occupation-ambiguous
#
# Inputs (in this folder): x28_Occupation_to_AVAM.xlsx, q93_isco.csv
# Outputs (in this folder): fig_ads_over_time.pdf, fig_isco3_count_over_time.pdf
#
# Run with this folder as the working directory. Needs DB access to
# archivedb.kof.ethz.ch with PG_USER / PG_PASSWORD set.

# Setup ----
.libPaths(c("../../../../lib", .libPaths()))
suppressMessages({
  library(data.table); library(dplyr); library(dbplyr)
  library(ggplot2);    library(readxl)
  library(DBI);        library(RPostgres)
})

WIN_START <- "2018-11-01"
WIN_END   <- "2025-12-01"     # half-open: [WIN_START, WIN_END)
MAX_JOBS  <- 5L

# Database pull ----
con <- dbConnect(Postgres(), host = "archivedb.kof.ethz.ch", db = "nrp77",
                 user = Sys.getenv("PG_USER"), password = Sys.getenv("PG_PASSWORD"))

adv  <- tbl(con, in_schema("x28", "advertisements"))
meta <- tbl(con, in_schema("x28", "advertisement_metadata"))
pos  <- tbl(con, in_schema("x28", "advertisement_positions"))

appr <- pos |> filter(position == "APPRENTICE") |>
  select(advertisement_id) |> distinct()

adv_win <- adv |>
  filter(created >= WIN_START, created < WIN_END) |>
  anti_join(appr, by = c("id" = "advertisement_id"))

ads <- adv_win |>
  select(id, created, duplicategroup) |>
  collect() |> as.data.table()

ad_jobs <- adv_win |>
  inner_join(meta |> filter(type == "JOB") |> select(advertisement_id, metadata_id),
             by = c("id" = "advertisement_id")) |>
  select(id, jobid = metadata_id) |>
  collect() |> as.data.table()

dbDisconnect(con)

# Deduplicate repostings (keep earliest per duplicategroup) ----
ads[, month := as.Date(format(created, "%Y-%m-01"))]
setorder(ads, duplicategroup, created)
ads <- rbind(
  unique(ads[!is.na(duplicategroup)], by = "duplicategroup"),
  ads[is.na(duplicategroup)]
)

# Occupation merge: job id -> AVAM -> ISCO-4 -> ISCO-3 ----
ad_jobs[, jobid := as.numeric(jobid)]
ad_jobs <- ad_jobs[id %in% ads$id]

jobs_by_id <- ad_jobs[, .N, by = id]
ad_jobs <- ad_jobs[!id %in% jobs_by_id[N > MAX_JOBS, id]]   # drop ambiguous ads

x28toavam <- as.data.table(readxl::read_excel("x28_Occupation_to_AVAM.xlsx"))
setnames(x28toavam, 1:5, c("id_x28", "name_x28", "avam", "name_avam", "drop"))
x28toavam[, drop := NULL]

q93_to_isco <- fread("q93_isco.csv")                        # cols: avam, isco
x28toavam   <- x28toavam[avam %in% q93_to_isco$avam]

id_to_isco <- merge(ad_jobs, x28toavam[, .(id_x28, avam)],
                    by.x = "jobid", by.y = "id_x28", all.x = TRUE, allow.cartesian = TRUE)
id_to_isco <- id_to_isco[!is.na(avam)]
id_to_isco <- merge(id_to_isco, q93_to_isco, by = "avam",
                    all.x = TRUE, allow.cartesian = TRUE)
id_to_isco <- unique(id_to_isco[, .(id, isco4 = as.numeric(isco))])
id_to_isco[, isco3 := isco4 %/% 10L]

# Monthly series ----
ads_ts  <- ads[, .(n_ads = .N), by = month]
isco_ts <- merge(unique(id_to_isco[, .(id, isco3)]), ads[, .(id, month)], by = "id")[
  , .(n_isco3 = uniqueN(isco3)), by = month]

ts <- merge(ads_ts, isco_ts, by = "month", all = TRUE)
setorder(ts, month)
ts[is.na(n_isco3), n_isco3 := 0L]

# Plots (count axis starts at 0) ----
p_ads <- ggplot(ts, aes(month, n_ads)) +
  geom_line() +
  scale_y_continuous(limits = c(0, NA), labels = scales::comma) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(x = NULL, y = "Vacancies created per month (deduplicated)")
ggsave("fig_ads_over_time.pdf", p_ads, width = 9, height = 4.5)

p_isco <- ggplot(ts, aes(month, n_isco3)) +
  geom_line() +
  scale_y_continuous(limits = c(0, NA)) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(x = NULL, y = "Distinct ISCO-3 occupations per month")
ggsave("fig_isco3_count_over_time.pdf", p_isco, width = 9, height = 4.5)
