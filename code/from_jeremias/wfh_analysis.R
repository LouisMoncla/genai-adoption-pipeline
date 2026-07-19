# Work-from-home in Swiss vacancies (x28): trend + AI-exposure correlation ----
#
# One script, one DB connection, three artifacts plus a diagnostics table:
#   (1) WFH share over time — composition of the `homeoffice` boolean
#       (TRUE / FALSE / missing), stacked shares per Nov->Oct "vacancy year"
#       starting November 2018.
#   (2) ISCO-08 3-digit WFH share vs. AI exposure — pooling ads created
#       Nov 2021 -> Oct 2022, correlated with the ISCO-3 GPT-4 exposure measure
#       (gpt4_beta, Eloundou et al. 2024) in a scatter with an OLS fit, plus a
#       top-10 / bottom-10 table of occupations by WFH share.
# (`homeoffice` is the literal DB column; everything user-facing says WFH.)
#
# MUST be run on the KOF server: needs DB access to archivedb.kof.ethz.ch and
# PG_USER / PG_PASSWORD env vars. The WFH indicator is only meaningful there.
#
# Lazy dbplyr throughout: only small summaries / one-year ad columns are
# collect()ed; the full advertisements table never leaves Postgres.
#
# Occupation mapping mirrors occupational-mobility/analysis/1_prepare_data/
# import_x28_ads.R:
#   ad -> advertisement_metadata(type='JOB').metadata_id (= x28 job id)
#      -> x28_Occupation_to_AVAM.xlsx (job id -> AVAM)
#      -> q93_isco.csv (AVAM -> ISCO-08 4-digit);  ISCO-3 = floor(ISCO-4 / 10)

# Setup ----
source("analysis/__config/_main.R")

suppressMessages({
  library(DBI); library(RPostgres); library(dbplyr)
})

# Configuration ----
START_YEAR  <- 2018L                   # first Nov->Oct window for the trend
WIN_START   <- "2021-11-01"            # ISCO-3 correlation: pooled window start (Nov->Oct year)
WIN_END     <- "2022-11-01"            # half-open window [START, END) = Nov 2021–Oct 2022
EXPOSURE    <- "gpt4_beta"             # AI-exposure column in indices_eloundou_3dig (ISCO-3)
MIN_CODED   <- 50L                     # min #ads with non-missing WFH per ISCO-3
SHARE_DENOM <- "coded"                 # "coded" = TRUE/(TRUE+FALSE); "all" = TRUE/N
MAX_JOBS    <- 5L                      # drop ads with > this many job ids (ambiguous)
OUT_FOLDER  <- "active/x28"            # relative to analysis/ (for save_plot)

# Database pulls (single connection) ----
con <- dbConnect(Postgres(), host = "archivedb.kof.ethz.ch", db = "nrp77",
                 user = Sys.getenv("PG_USER"), password = Sys.getenv("PG_PASSWORD"))

adv  <- tbl(con, dbplyr::in_schema("x28", "advertisements"))
meta <- tbl(con, dbplyr::in_schema("x28", "advertisement_metadata"))
pos  <- tbl(con, dbplyr::in_schema("x28", "advertisement_positions"))

# Pull 1: year x month x homeoffice counts for the trend (tiny summary)
wfh_ym <- adv |>
  filter(created >= "2018-11-01") |>
  mutate(yr = sql("EXTRACT(YEAR FROM created)::int"),
         mo = sql("EXTRACT(MONTH FROM created)::int")) |>
  count(yr, mo, homeoffice) |>
  collect() |> as.data.table()

# Pull 2: ad-level fields + ad->jobid for the pooled ISCO-3 window
appr <- pos |> filter(position == "APPRENTICE") |>      # apprenticeships to exclude
  select(advertisement_id) |> distinct()

adv_win <- adv |>
  filter(created >= WIN_START, created < WIN_END) |>
  anti_join(appr, by = c("id" = "advertisement_id"))

ads <- adv_win |>                                       # one row per ad (minimal cols)
  select(id, created, homeoffice, duplicategroup) |>    # created only for earliest-dedup
  collect() |> as.data.table()

ad_jobs <- adv_win |>                                   # one row per JOB metadata entry
  inner_join(meta |> filter(type == "JOB") |> select(advertisement_id, metadata_id),
             by = c("id" = "advertisement_id")) |>
  select(id, jobid = metadata_id) |>                    # minimal cols
  collect() |> as.data.table()

dbDisconnect(con)

# Trend: WFH share over time (stacked Nov->Oct shares) ----

# Assign Nov->Oct windows: months Nov-Dec -> year Y, Jan-Oct -> year (Y-1).
wfh_ym[, win := fifelse(mo >= 11L, yr, yr - 1L)]
wfh_ym <- wfh_ym[win >= START_YEAR]

# Keep only complete windows (all 12 months present) for comparable shares
months_per_win <- wfh_ym[, .(n_months = uniqueN(paste(yr, mo))), by = win]
complete_wins  <- months_per_win[n_months >= 12L, win]
dropped <- sort(setdiff(unique(wfh_ym$win), complete_wins))
if (length(dropped))
  message("Dropping incomplete Nov->Oct window(s): ", paste(dropped, collapse = ", "))
wfh_ym <- wfh_ym[win %in% complete_wins]

wfh_ym[, wfh_cat := fcase(
  homeoffice == TRUE,  "Work from home mentioned",
  homeoffice == FALSE, "Not mentioned",
  default              = "Not coded")]                  # NA / missing

wfh_win <- wfh_ym[, .(n = sum(n)), by = .(win, wfh_cat)]
wfh_win[, share := n / sum(n), by = win]
wfh_win[, wfh_cat := factor(wfh_cat,
        levels = c("Not coded", "Not mentioned", "Work from home mentioned"))]
wfh_win[, win_lab := paste0("Nov ", win, "–\nOct ", win + 1L)]
wfh_win[, win_lab := factor(win_lab, levels = unique(win_lab[order(win)]))]
setorder(wfh_win, win, wfh_cat)

fwrite(wfh_win[, .(window_start = win, window = gsub("\n", " ", as.character(win_lab)),
                   category = as.character(wfh_cat), n, share)],
       "analysis/active/x28/wfh_share_over_time.csv")

fill_cols <- c(
  "Work from home mentioned" = unname(COLOURS["green"]),
  "Not mentioned"            = unname(COLOURS["bluegrey"]),
  "Not coded"                = unname(COLOURS["grey"]))

p_trend <- ggplot(wfh_win, aes(x = win_lab, y = share, fill = wfh_cat)) +
  geom_col(width = 0.75) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.02))) +
  scale_fill_manual(values = fill_cols, name = NULL,
                    guide = guide_legend(reverse = TRUE)) +
  labs(x = NULL, y = "Share of vacancies", title = NULL, subtitle = NULL) +
  mytheme +
  theme(text = element_text(family = "serif"),
        legend.position = "bottom",
        axis.text.x = element_text(size = rel(0.8)))
p_trend
save_plot(p_trend, OUT_FOLDER, "fig_wfh_share_over_time", width = 9, height = 5)

# Occupation panel: ISCO-3 WFH share vs. AI exposure ----
ad_jobs[, jobid := as.numeric(jobid)]

# Deduplicate ads by duplicategroup (keep earliest), guarding NA groups
setorder(ads, duplicategroup, created)
ads <- rbind(
  unique(ads[!is.na(duplicategroup)], by = "duplicategroup"),
  ads[is.na(duplicategroup)]
)
ad_jobs <- ad_jobs[id %in% ads$id]

# Diagnostic: overall WFH-flag composition in the pooling window
n_w  <- nrow(ads); n_na <- ads[is.na(homeoffice), .N]
n_t  <- ads[homeoffice == TRUE, .N]; n_f <- ads[homeoffice == FALSE, .N]
cat("\nWFH flag in pooling window ", WIN_START, " .. ", WIN_END, ":\n", sep = "")
cat("  ads (deduped, non-apprentice): ", n_w, "\n", sep = "")
cat("  missing (NA): ", n_na, " (", round(100 * n_na / n_w, 2), "%)\n", sep = "")
cat("  TRUE:  ", n_t, " (", round(100 * n_t / n_w, 2), "%)\n", sep = "")
cat("  FALSE: ", n_f, " (", round(100 * n_f / n_w, 2), "%)\n", sep = "")

# Drop ads with too many job ids (ambiguous occupation)
jobs_by_id <- ad_jobs[, .N, by = id]
ad_jobs <- ad_jobs[!id %in% jobs_by_id[N > MAX_JOBS, id]]

# Occupation mapping: job id -> AVAM -> ISCO-08 4-digit -> 3-digit
x28toavam <- as.data.table(readxl::read_excel(
  paste0(backend_dir, "help-files/x28_Occupation_to_AVAM.xlsx")))
setnames(x28toavam, 1:5, c("id_x28", "name_x28", "avam", "name_avam", "drop"))
x28toavam[, drop := NULL]

q93_to_isco <- fread(paste0(backend_dir, "help-files/q93_isco.csv"))   # cols: avam, isco
x28toavam   <- x28toavam[avam %in% q93_to_isco$avam]

id_to_isco <- merge(ad_jobs, x28toavam[, .(id_x28, avam)],
                    by.x = "jobid", by.y = "id_x28", all.x = TRUE, allow.cartesian = TRUE)
cat("Job ids without AVAM match:", round(id_to_isco[, mean(is.na(avam)) * 100], 2), "%\n")
id_to_isco <- id_to_isco[!is.na(avam)]
id_to_isco <- merge(id_to_isco, q93_to_isco, by = "avam", all.x = TRUE, allow.cartesian = TRUE)
id_to_isco <- unique(id_to_isco[, .(id, isco4 = as.numeric(isco))])
id_to_isco[, isco3 := isco4 %/% 10L]

# Attach ISCO-3 to ad-level WFH indicator
ads_isco <- merge(ads[, .(id, homeoffice)], id_to_isco[, .(id, isco3)], by = "id")
cat("Ad-ISCO rows:", nrow(ads_isco), "| unique ISCO-3:", uniqueN(ads_isco$isco3), "\n")

# WFH share per ISCO-3 (+ save indicator to parquet) ----
wfh_by_isco3 <- ads_isco[, .(
  n_total = .N,
  n_true  = sum(homeoffice == TRUE,  na.rm = TRUE),
  n_false = sum(homeoffice == FALSE, na.rm = TRUE)
), by = isco3]
wfh_by_isco3[, n_coded := n_true + n_false]

# Diagnostic: ISCO-3 cells below the MIN_CODED threshold (dropped)
n_all_cells   <- nrow(wfh_by_isco3)
n_below       <- wfh_by_isco3[n_coded < MIN_CODED, .N]
n_below_coded <- wfh_by_isco3[n_coded < MIN_CODED, sum(n_coded)]
n_total_coded <- wfh_by_isco3[, sum(n_coded)]
cat("\nISCO-3 occupations: ", n_all_cells, " total | ", n_below, " with < ",
    MIN_CODED, " coded ads (", round(100 * n_below / n_all_cells, 1), "%) -> dropped\n", sep = "")
cat("  coded ads in dropped cells: ", n_below_coded, " of ", n_total_coded, " total coded\n", sep = "")

wfh_by_isco3 <- wfh_by_isco3[n_coded >= MIN_CODED]
wfh_by_isco3[, wfh_share := if (SHARE_DENOM == "coded") n_true / n_coded
                            else n_true / n_total]
cat("ISCO-3 cells with >=", MIN_CODED, "coded ads:", nrow(wfh_by_isco3), "\n")

# Occupation labels (project ISCO labels table; `labels$isco` matched as a
# 3-digit code; cells without a 3-digit label keep the numeric code)
lab3 <- unique(labels[, .(isco3 = as.numeric(isco), occ = as.character(isconame_clean))])
wfh_by_isco3 <- merge(wfh_by_isco3, lab3, by = "isco3", all.x = TRUE)
wfh_by_isco3[is.na(occ), occ := paste0("ISCO ", isco3)]
setorder(wfh_by_isco3, isco3)

save_data(wfh_by_isco3, "x28_wfh_share_isco3")          # arrow/parquet project store

# AI exposure: directly-saved 3-digit Eloundou index ----
# Employment-weighted aggregation of the 4-digit measures (`isco` is the
# 3-digit code) — not a plain mean.
elo3 <- read_data("indices_eloundou_3dig", format = ".csv")
setDT(elo3)
exp3 <- elo3[, .(isco3 = as.numeric(isco), exposure = get(EXPOSURE))]
exp3 <- exp3[!is.na(isco3) & is.finite(exposure)]

d <- merge(wfh_by_isco3, exp3, by = "isco3")
d <- d[is.finite(exposure)]
cat("ISCO-3 cells matched to exposure:", nrow(d), "\n")

# OLS + scatter ----
m  <- feols(wfh_share ~ exposure, data = d)
ct <- m$coeftable["exposure", ]
slope <- ct[["Estimate"]]; se <- ct[["Std. Error"]]; pval <- ct[["Pr(>|t|)"]]
star  <- fcase(pval < 0.01, "***", pval < 0.05, "**", pval < 0.1, "*", default = "")
r2v   <- unname(fixest::r2(m, "r2"))
annot <- sprintf("Slope = %.3f%s  (SE %.3f)\nR² = %.2f,  N = %d ISCO-3",
                 slope, star, se, r2v, nrow(d))
cat("\n", annot, "\n", sep = "")

xlab <- "AI exposure (Eloundou et al. 2024, GPT-4 β, ISCO-3)"
ylab <- "Work-from-home share"

p_scatter <- ggplot(d, aes(x = exposure, y = wfh_share)) +
  geom_point(aes(size = n_coded), alpha = 0.45, colour = unname(COLOURS["blue"])) +
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE,
              colour = unname(COLOURS["signalred"]),
              fill   = unname(COLOURS["signalred"]), alpha = 0.15, linewidth = 1) +
  annotate("text", x = -Inf, y = Inf, hjust = -0.06, vjust = 1.25,
           label = annot, family = "serif", size = 4.2) +
  scale_size_continuous(name = "Coded vacancies", range = c(0.8, 6), labels = scales::comma) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(x = xlab, y = ylab, title = NULL, subtitle = NULL) +
  mytheme +
  theme(text = element_text(family = "serif"), legend.position = "right")
p_scatter
save_plot(p_scatter, OUT_FOLDER, "fig_wfh_share_vs_ai_exposure_isco3", width = 7.5, height = 5.2)

# Top-10 / bottom-10 occupations by WFH share ----
setorder(d, -wfh_share)
top10 <- d[1:min(10L, nrow(d))]
bot10 <- d[order(wfh_share)][1:min(10L, nrow(d))]

fwrite(rbind(cbind(panel = "Top 10",    top10[, .(isco3, occ, wfh_share, n_coded, exposure)]),
             cbind(panel = "Bottom 10", bot10[, .(isco3, occ, wfh_share, n_coded, exposure)])),
       "analysis/active/x28/topbottom10_wfh_share_isco3.csv")

# Bare booktabs tabular (no caption/label/notes — those live in the paper)
esc <- function(x) gsub("%", "\\\\%", gsub("&", "\\\\&", gsub("_", "\\\\_", x)))
fmt_row <- function(rank, r)
  sprintf("%d & %d & %s & %.1f\\%% & %s \\\\",
          rank, r$isco3, esc(r$occ), 100 * r$wfh_share,
          formatC(r$n_coded, format = "d", big.mark = ","))

rows_top <- vapply(seq_len(nrow(top10)), function(i) fmt_row(i, top10[i]), character(1))
rows_bot <- vapply(seq_len(nrow(bot10)), function(i) fmt_row(i, bot10[i]), character(1))

tex <- c(
  "\\begin{tabular}{rllrr}",
  "\\toprule",
  " & ISCO-3 & Occupation & WFH share & Coded vac. \\\\",
  "\\midrule",
  "\\multicolumn{5}{l}{\\textit{Panel A: Highest work-from-home share}} \\\\",
  "\\midrule",
  rows_top,
  "\\midrule",
  "\\multicolumn{5}{l}{\\textit{Panel B: Lowest work-from-home share}} \\\\",
  "\\midrule",
  rows_bot,
  "\\bottomrule",
  "\\end{tabular}")
writeLines(tex, "analysis/active/x28/topbottom10_wfh_share_isco3.tex")

# Diagnostics table (for note-writing, not the paper) ----
# Small metric/value table the note-writing agent can read.
diag <- data.table(
  metric = c(
    "window_start", "window_end", "share_denominator", "exposure_measure", "min_coded_threshold",
    "n_ads_window", "n_missing", "pct_missing", "n_true", "pct_true", "n_false", "pct_false",
    "n_isco3_total", "n_isco3_below_threshold", "pct_isco3_below_threshold",
    "coded_ads_in_dropped_cells", "total_coded_ads",
    "n_isco3_in_regression", "ols_slope", "ols_se", "ols_pvalue", "ols_r2"),
  value = c(
    WIN_START, WIN_END, SHARE_DENOM, EXPOSURE, MIN_CODED,
    n_w, n_na, round(100 * n_na / n_w, 2), n_t, round(100 * n_t / n_w, 2),
    n_f, round(100 * n_f / n_w, 2),
    n_all_cells, n_below, round(100 * n_below / n_all_cells, 1),
    n_below_coded, n_total_coded,
    nrow(d), round(slope, 4), round(se, 4), signif(pval, 3), round(r2v, 3)))
fwrite(diag, "analysis/active/x28/wfh_diagnostics.csv")

cat("\nDone. Outputs:\n",
    "  parquet (data store): x28_wfh_share_isco3\n",
    "  analysis/", OUT_FOLDER, "/fig_wfh_share_over_time.pdf\n",
    "  analysis/", OUT_FOLDER, "/fig_wfh_share_vs_ai_exposure_isco3.pdf\n",
    "  analysis/", OUT_FOLDER, "/wfh_share_over_time.csv\n",
    "  analysis/", OUT_FOLDER, "/topbottom10_wfh_share_isco3.csv / .tex\n",
    "  analysis/", OUT_FOLDER, "/wfh_diagnostics.csv\n", sep = "")
