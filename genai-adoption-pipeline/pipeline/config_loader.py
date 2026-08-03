"""
Configuration loader for the GenAI Adoption Pipeline.
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

import yaml


@dataclass
class ColumnMapping:
    """Abstracted column names for the dataset."""
    date: str
    content: str
    language: str          # x28's own detected-language flag (x28: "language")
    locations: str
    job_id: str           # Top-level posting dedup key (x28: "duplicate_group")
    # Nested company fields
    company_struct: str
    company_id: str
    company_name: str
    company_is_recruiter: str
    company_metadata: str
    company_size_struct: str
    company_size_id: str
    company_size_name: str


@dataclass
class PipelineConfig:
    """Typed configuration for the GenAI Adoption Pipeline."""
    data_path: Path
    col: ColumnMapping
    date_range_start: Optional[date]
    date_range_end: Optional[date]
    country_filter: Union[str, list[str]]
    allowed_languages: Optional[list]
    exclude_recruiters: bool
    exclude_micro_enterprises: bool
    micro_enterprise_id: str
    sanity_ci_alpha: float          # kept for backwards compatibility / fallback
    sanity_alpha_sme: float
    sanity_alpha_medium: float
    sanity_alpha_large: float
    
    # Validation Parameters
    pre_period_end_year: int
    validation_year: int
    dhs_group1_threshold: float
    dhs_group2_threshold: float
    dhs_group3_threshold: float
    ratio_group2_threshold: float
    ratio_group3_threshold: float
    g1_occupation_coverage_threshold: float
    min_g1_required: int
    min_g2_for_rule2: int
    min_g2g3_for_rule3: int
    enforce_occupation_filter: bool

    adoption_threshold_ads: int
    min_postings_for_intensity: int
    min_occupation_intensity: float
    output_dir: Path


def load_config(config_path: str = "pipeline/config.yaml") -> PipelineConfig:
    """Load and validate pipeline configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(raw).__name__}")

    # --- column_mapping ---
    m = raw.get("column_mapping", {})
    col = ColumnMapping(
        date=m.get("date", "tst_created"),
        content=m.get("content", "content_clean"),
        language=m.get("language", "language"),
        locations=m.get("locations", "locations"),
        job_id=m.get("job_id", "id"),
        company_struct=m.get("company_struct", "company"),
        company_id=m.get("company_id", "id"),
        company_name=m.get("company_name", "name"),
        company_is_recruiter=m.get("company_is_recruiter", "is_recruiter"),
        company_metadata=m.get("company_metadata", "metadata"),
        company_size_struct=m.get("company_size_struct", "size"),
        company_size_id=m.get("company_size_id", "id"),
        company_size_name=m.get("company_size_name", "name")
    )

    data_path = Path(raw.get("data_path", "data/"))
    date_range_start = _parse_optional_date(raw.get("date_range_start"), "date_range_start")
    date_range_end = _parse_optional_date(raw.get("date_range_end"), "date_range_end")

    country_filter = raw.get("country_filter", "CH")
    allowed_languages = raw.get("allowed_languages", None)
    exclude_recruiters = raw.get("exclude_recruiters", True)
    exclude_micro = raw.get("exclude_micro_enterprises", True)
    micro_id = str(raw.get("micro_enterprise_id", "57000001"))
    ci_alpha        = float(raw.get("sanity_ci_alpha",    0.01))
    alpha_sme       = float(raw.get("sanity_alpha_sme",    0.005))
    alpha_medium    = float(raw.get("sanity_alpha_medium", 0.002))
    alpha_large     = float(raw.get("sanity_alpha_large",  0.001))
    
    pre_period_end_year = int(raw.get("pre_period_end_year", 2021))
    validation_year = int(raw.get("validation_year", 2025))
    dhs_g1 = float(raw.get("dhs_group1_threshold", 1.95))
    dhs_g2 = float(raw.get("dhs_group2_threshold", 1.72))
    dhs_g3 = float(raw.get("dhs_group3_threshold", 1.0))
    rat_g2 = float(raw.get("ratio_group2_threshold", 11.75))
    rat_g3 = float(raw.get("ratio_group3_threshold", 20.0))
    g1_occ_thresh = float(raw.get("g1_occupation_coverage_threshold", 0.80))
    min_g1 = int(raw.get("min_g1_required", 1))
    min_g2_rule2 = int(raw.get("min_g2_for_rule2", 1))
    min_g2_rule3 = int(raw.get("min_g2g3_for_rule3", 3))
    enforce_occ = bool(raw.get("enforce_occupation_filter", True))

    adoption_thresh = int(raw.get("adoption_threshold_ads", 1))
    min_postings_intensity = int(raw.get("min_postings_for_intensity", 5))
    min_occ_intensity = float(raw.get("min_occupation_intensity", 0.02))

    output_dir = Path(raw.get("output_dir", "output/"))

    return PipelineConfig(
        data_path=data_path, col=col, date_range_start=date_range_start,
        date_range_end=date_range_end, country_filter=country_filter,
        allowed_languages=allowed_languages,
        exclude_recruiters=exclude_recruiters, exclude_micro_enterprises=exclude_micro,
        micro_enterprise_id=micro_id, sanity_ci_alpha=ci_alpha,
        sanity_alpha_sme=alpha_sme, sanity_alpha_medium=alpha_medium,
        sanity_alpha_large=alpha_large,
        pre_period_end_year=pre_period_end_year,
        validation_year=validation_year,
        dhs_group1_threshold=dhs_g1,
        dhs_group2_threshold=dhs_g2,
        dhs_group3_threshold=dhs_g3,
        ratio_group2_threshold=rat_g2,
        ratio_group3_threshold=rat_g3,
        g1_occupation_coverage_threshold=g1_occ_thresh,
        min_g1_required=min_g1,
        min_g2_for_rule2=min_g2_rule2,
        min_g2g3_for_rule3=min_g2_rule3,
        enforce_occupation_filter=enforce_occ,
        adoption_threshold_ads=adoption_thresh,
        min_postings_for_intensity=min_postings_intensity,
        min_occupation_intensity=min_occ_intensity,
        output_dir=output_dir,
    )


def _parse_optional_date(value, field_name: str) -> Optional[date]:
    if value is None: return None
    if isinstance(value, datetime): return value.date()
    if isinstance(value, date): return value
    if isinstance(value, str):
        try: return date.fromisoformat(value)
        except ValueError: raise ValueError(f"'{field_name}' must be ISO YYYY-MM-DD")
    raise ValueError(f"'{field_name}' must be date or null")
