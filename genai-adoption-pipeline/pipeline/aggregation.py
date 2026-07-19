"""
Phase III: Firm-level aggregation, AI adoption index, and multi-dimensional breakdowns.
"""

import logging
from pathlib import Path

import polars as pl

from pipeline.config_loader import PipelineConfig

logger = logging.getLogger(__name__)


def aggregate_results(config: PipelineConfig) -> dict[str, pl.DataFrame]:
    """
    Execute Phase III: Aggregation of scored postings.
    
    Architecture Note: Uses `.collect(streaming=True)` on the fully chunked
    parquet partitions. This invokes the Polars out-of-core streaming engine,
    allowing complex multi-dimensional aggregation across 20M rows purely on-disk.
    """
    output_dir = Path(config.output_dir)
    agg_dir = output_dir / "aggregation"
    agg_dir.mkdir(parents=True, exist_ok=True)

    # Scored data is now in partitioned directory
    data_path = output_dir / "scored_data"
    logger.info("=== Phase III: Aggregation (Big Data Optimized) ===")
    
    # Scan all partitions
    lf = pl.scan_parquet(data_path / "year=*" / "*.parquet")

    # Add year if needed (should already be there from Phase I unnesting)
    lf = lf.with_columns(
        pl.col(config.col.date).dt.year().alias("year")
    )

    results = {}

    # 1. Firm-level aggregation
    logger.info("  Aggregating firm-level metrics...")
    results["firm_level"] = _aggregate_firm_level(lf, config).collect(streaming=True)
    results["firm_level"].write_parquet(agg_dir / "firm_level.parquet")

    # 2. Time-series aggregation
    logger.info("  Aggregating overall time-series...")
    results["time_series"] = _aggregate_time_series(lf).collect(streaming=True)
    results["time_series"].write_parquet(agg_dir / "time_series.parquet")

    # 3. Adoption Index
    logger.info("  Aggregating adoption index...")
    results["adoption_index"] = _aggregate_adoption_index(lf, config).collect(streaming=True)
    results["adoption_index"].write_parquet(agg_dir / "adoption_index.parquet")

    # 4. Intensity Analysis
    logger.info("  Aggregating firm intensity analysis...")
    results["firm_intensity"] = _aggregate_intensity_analysis(lf, config).collect(streaming=True)
    results["firm_intensity"].write_parquet(agg_dir / "firm_intensity_top20.parquet")

    # 5. Industry-level
    if "_industry" in lf.collect_schema().names():
        logger.info("  Aggregating industry breakdowns...")
        results["industry_level"] = _aggregate_industry_level(lf, config).collect(streaming=True)
        results["industry_level"].write_parquet(agg_dir / "industry_level.parquet")
    else:
        logger.warning("  Skipping industry breakdowns: '_industry' column not found in data.")
        # Create an empty dummy DataFrame to prevent downstream dependency issues
        results["industry_level"] = pl.DataFrame({"_industry": [], "total_postings": [], "genai_postings": [], "genai_percentage": []})

    # 5b. Industry x year — AI-ad frequency per industry over time
    if "_industry" in lf.collect_schema().names():
        logger.info("  Aggregating industry-by-year breakdowns...")
        results["industry_time_series"] = _aggregate_industry_time(lf, config).collect(streaming=True)
        results["industry_time_series"].write_parquet(agg_dir / "industry_time_series.parquet")
    else:
        results["industry_time_series"] = pl.DataFrame({"_industry": [], "year": [], "total_postings": [], "genai_postings": [], "genai_percentage": []})

    # 6. Occupation-level (New)
    logger.info("  Aggregating occupation breakdowns...")
    results["occupation_level"] = _aggregate_occupation_level(lf).collect(streaming=True)
    results["occupation_level"].write_parquet(agg_dir / "occupation_level.parquet")

    # 7. Keyword Time-Series
    logger.info("  Aggregating keyword trends...")
    # Keyword TS needs special handling due to list explosion
    df_kw = lf.select(["matched_keywords", "year", "genai_flag", "flag_rule_1", "flag_rule_2", "flag_rule_3", "flag_rule_copilot"]).collect(streaming=True)
    results["keyword_ts"] = _aggregate_keyword_time_series_df(df_kw)
    results["keyword_ts"].write_parquet(agg_dir / "keyword_time_series.parquet")

    logger.info(f"All aggregations saved to {agg_dir}")
    return results


def _aggregate_adoption_index(lf: pl.LazyFrame, config: PipelineConfig) -> pl.LazyFrame:
    thresh = config.adoption_threshold_ads
    
    # Ensure consistent year dtype (i64) across all joins
    lf = lf.with_columns(pl.col("year").cast(pl.Int64))
    
    # 1. Get the first adoption year for each firm
    # A firm is an adopter in a year if its genai_flag sum >= threshold
    firm_adoption = lf.group_by(["_firm_id", "year"]).agg(
        pl.col("genai_flag").sum().alias("genai_count")
    ).filter(pl.col("genai_count") >= thresh)
    
    first_adoption = firm_adoption.group_by("_firm_id").agg(
        pl.col("year").min().alias("first_year")
    )
    
    # 2. Get total unique firms that posted in each year (for the denominator)
    total_firms_per_year = lf.select(["year", "_firm_id"]).unique().group_by("year").agg(
        pl.len().alias("total_firms_posting")
    )
    
    # 3. Create a cumulative count of adopters
    # We need to count how many firms adopted for the first time in each year
    new_adopters_per_year = first_adoption.group_by("first_year").agg(
        pl.len().alias("new_adopters")
    ).rename({"first_year": "year"}).sort("year")
    
    # Fill in missing years with 0 new adopters to ensure cumulative math works
    all_years = pl.DataFrame({"year": list(range(2012, 2026))}, schema={"year": pl.Int64}).lazy()
    new_adopters_per_year = all_years.join(new_adopters_per_year, on="year", how="left").fill_null(0)
    
    # Calculate cumulative sum
    adoption_index = new_adopters_per_year.with_columns(
        pl.col("new_adopters").cum_sum().alias("adopting_firms_count")
    )
    
    # 4. Join with total firms to get share
    adoption_index = adoption_index.join(total_firms_per_year, on="year", how="left").fill_null(0)
    
    adoption_index = adoption_index.with_columns(
        (pl.col("adopting_firms_count").cast(pl.Float64) / pl.col("total_firms_posting") * 100).alias("adoption_share")
    ).sort("year")
    
    return adoption_index.lazy()


def _aggregate_intensity_analysis(lf: pl.LazyFrame, config: PipelineConfig) -> pl.LazyFrame:
    firm_yearly = lf.group_by(["year", "_firm_id"]).agg(
        pl.col("_company_name").first().alias("company_name"),
        pl.len().alias("total_postings"),
        pl.col("genai_flag").sum().alias("genai_postings")
    )
    # Filter by minimum postings to avoid 1/1 = 100% outliers
    firm_yearly = firm_yearly.filter(pl.col("total_postings") >= config.min_postings_for_intensity)
    
    firm_yearly = firm_yearly.with_columns(
        (pl.col("genai_postings").cast(pl.Float64) / pl.col("total_postings") * 100).alias("genai_intensity")
    )
    firm_yearly = firm_yearly.filter(pl.col("genai_postings") > 0)
    return firm_yearly.sort(["year", "genai_intensity"], descending=[False, True]).group_by("year").head(20)


def _aggregate_firm_level(lf: pl.LazyFrame, config: PipelineConfig) -> pl.LazyFrame:
    firm = lf.group_by("_firm_id").agg(
        pl.col("_company_name").first().alias("company_name"),
        pl.col("_size_name").first().alias("size_category"),
        pl.len().alias("total_postings"),
        pl.col("genai_flag").sum().alias("genai_postings"),
        pl.col("group1_match").sum().alias("group1_count"),
        pl.col("group2_match").sum().alias("group2_count"),
    )
    firm = firm.with_columns(
        (pl.col("genai_postings").cast(pl.Float64) / pl.col("total_postings")).alias("genai_rate")
    )
    return firm.sort("genai_rate", descending=True)


def _aggregate_time_series(lf: pl.LazyFrame) -> pl.LazyFrame:
    ts = lf.group_by("year").agg(
        pl.len().alias("total_postings"),
        pl.col("genai_flag").sum().alias("genai_postings"),
    ).sort("year")
    ts = ts.with_columns(
        (pl.col("genai_postings").cast(pl.Float64) / pl.col("total_postings") * 100).alias("genai_percentage")
    )
    return ts


def _aggregate_occupation_level(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate GenAI adoption metrics by occupation and year."""
    # 1. Extract first occupation name from the list
    lf = lf.with_columns(
        pl.col("occupations").list.eval(
            pl.element().struct.field("name")
        ).list.first().alias("occupation")
    )
    
    # 2. Group by occupation and year
    occ = lf.group_by(["year", "occupation"]).agg(
        pl.len().alias("total_postings"),
        pl.col("genai_flag").sum().alias("genai_postings"),
    )
    
    # 3. Calculate percentage
    occ = occ.with_columns(
        (pl.col("genai_postings").cast(pl.Float64) / pl.col("total_postings") * 100).alias("genai_percentage")
    )
    
    return occ.sort(["year", "genai_postings"], descending=[False, True])


def _aggregate_keyword_time_series_df(df: pl.DataFrame) -> pl.DataFrame:
    kw_df = df.explode("matched_keywords")
    kw_df = kw_df.filter(pl.col("matched_keywords").is_not_null())
    return kw_df.group_by(["matched_keywords", "year"]).agg(
        pl.len().alias("match_count"),
        pl.col("genai_flag").cast(pl.Int64).sum().alias("flagged_count"),
        (pl.col("genai_flag") & pl.col("flag_rule_1")).cast(pl.Int64).sum().alias("flagged_g1"),
        (pl.col("genai_flag") & pl.col("flag_rule_2") & ~pl.col("flag_rule_1")).cast(pl.Int64).sum().alias("flagged_g2"),
        (pl.col("genai_flag") & pl.col("flag_rule_3") & ~pl.col("flag_rule_1") & ~pl.col("flag_rule_2")).cast(pl.Int64).sum().alias("flagged_g3"),
        (pl.col("genai_flag") & pl.col("flag_rule_copilot") & ~pl.col("flag_rule_1") & ~pl.col("flag_rule_2") & ~pl.col("flag_rule_3")).cast(pl.Int64).sum().alias("flagged_rc")
    ).sort(["matched_keywords", "year"])


def _aggregate_industry_level(lf: pl.LazyFrame, config: PipelineConfig) -> pl.LazyFrame:
    # Industry was pre-extracted during Phase I as _industry
    industry = lf.group_by("_industry").agg(
        pl.len().alias("total_postings"),
        pl.col("genai_flag").sum().alias("genai_postings"),
    ).sort("total_postings", descending=True)
    industry = industry.with_columns(
        (pl.col("genai_postings").cast(pl.Float64) / pl.col("total_postings") * 100).alias("genai_percentage")
    )
    return industry


def _aggregate_industry_time(lf: pl.LazyFrame, config: PipelineConfig) -> pl.LazyFrame:
    """Industry x year: frequency of AI job ads per industry over time.

    One row per (_industry, year) with total postings, GenAI-flagged postings,
    and the GenAI share (%). This is the table for "does industry X have a
    relatively high frequency of AI ads at a given time".
    """
    industry = lf.group_by(["_industry", "year"]).agg(
        pl.len().alias("total_postings"),
        pl.col("genai_flag").sum().alias("genai_postings"),
    )
    industry = industry.with_columns(
        (pl.col("genai_postings").cast(pl.Float64) / pl.col("total_postings") * 100).alias("genai_percentage")
    )
    return industry.sort(["_industry", "year"])

