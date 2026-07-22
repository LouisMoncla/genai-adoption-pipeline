"""
Quick visualizations straight from the exported Parquet files
(data/processed/x28_parquet/) - no pipeline phases needed, just aggregating
columns that already exist in the raw export.

Works on however many days have been exported so far - the export doesn't
need to be finished to run these.
"""

from pathlib import Path

import plotly.express as px
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "x28_parquet"
OUT_DIR = ROOT / "outputs"


def plot_postings_per_month(lf: pl.LazyFrame) -> None:
    monthly = (
        lf.select(pl.col("tst_created").str.slice(0, 7).alias("month"))  # "YYYY-MM"
        .group_by("month")
        .agg(pl.len().alias("postings"))
        .sort("month")
        .collect()
    )

    print(f"{monthly.height} months covered, {monthly['postings'].sum():,} postings total.")

    fig = px.bar(
        monthly, x="month", y="postings",
        title="Job postings per month (x28 dataset, as exported so far)",
        labels={"month": "", "postings": "Number of postings"},
    )
    fig.update_xaxes(tickangle=-45)

    out_path = OUT_DIR / "postings_per_month.html"
    fig.write_html(out_path)
    print(f"Saved to {out_path}")


def plot_homeoffice_share_per_month(lf: pl.LazyFrame) -> None:
    """Share of postings advertising home-office per month. Worth seeing
    alongside code/from_jeremias/wfh_analysis.R - same underlying question
    (remote-work trends in the Swiss labour market), here just as a plain
    descriptive check straight off the raw export, no modeling."""
    monthly = (
        lf.select(
            pl.col("tst_created").str.slice(0, 7).alias("month"),
            pl.col("has_homeoffice"),
        )
        .group_by("month")
        .agg(
            pl.len().alias("postings"),
            pl.col("has_homeoffice").sum().alias("homeoffice_postings"),
        )
        .with_columns((pl.col("homeoffice_postings") / pl.col("postings")).alias("share"))
        .sort("month")
        .collect()
    )

    fig = px.line(
        monthly, x="month", y="share", markers=True,
        title="Share of postings advertising home-office, per month",
        labels={"month": "", "share": "Share of postings"},
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickangle=-45)

    out_path = OUT_DIR / "homeoffice_share_per_month.html"
    fig.write_html(out_path)
    print(f"Saved to {out_path}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    lf = pl.scan_parquet(DATA_DIR / "x28_ads_*.parquet")

    plot_postings_per_month(lf)
    plot_homeoffice_share_per_month(lf)


if __name__ == "__main__":
    main()
