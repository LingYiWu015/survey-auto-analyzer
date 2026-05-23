"""Utility functions for data preprocessing and common operations."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def ensure_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert selected columns to numeric, coercing errors to NaN.

    For Likert scales with Chinese text (e.g., "非常满意"), also maps them to numbers.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue

        # Try direct numeric conversion first
        numeric = pd.to_numeric(df[col], errors="coerce")

        # If most values failed to convert, try mapping Chinese Likert
        if numeric.isna().mean() > 0.3:
            mapped = _map_likert_text(df[col])
            if mapped is not None:
                df[col] = mapped
                continue

        df[col] = numeric

    return df


def _map_likert_text(series: pd.Series) -> Optional[pd.Series]:
    """Try to map Chinese Likert scale text to 1-5 numbers.

    Returns mapped Series if successful, None otherwise.
    """
    text_to_num = {
        "非常不满意": 1, "不满意": 2, "一般": 3, "满意": 4, "非常满意": 5,
        "非常不同意": 1, "不同意": 2, "中立": 3, "同意": 4, "非常同意": 5,
        "非常不赞同": 1, "不赞同": 2, "一般": 3, "赞同": 4, "非常赞同": 5,
        "从不": 1, "很少": 2, "有时": 3, "经常": 4, "总是": 5,
        "非常低": 1, "较低": 2, "中等": 3, "较高": 4, "非常高": 5,
        "非常不重要": 1, "不重要": 2, "一般": 3, "重要": 4, "非常重要": 5,
        # English fallsbacks
        "strongly disagree": 1, "disagree": 2, "neutral": 3,
        "agree": 4, "strongly agree": 5,
        "very dissatisfied": 1, "dissatisfied": 2, "neutral": 3,
        "satisfied": 4, "very satisfied": 5,
    }

    norm_series = series.astype(str).str.strip()
    mapped = norm_series.map(text_to_num)

    if mapped.notna().mean() > 0.5:  # At least 50% mapped successfully
        return mapped
    return None


def drop_high_missing(
    df: pd.DataFrame, threshold: float = 0.3
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns with missing rate above threshold.

    Returns (cleaned_df, list_of_dropped_columns).
    """
    missing_rates = df.isna().mean()
    dropped = missing_rates[missing_rates > threshold].index.tolist()

    if dropped:
        logger.info(f"Dropped {len(dropped)} high-missing columns: {dropped}")
        df = df.drop(columns=dropped)

    return df, dropped


def onehot_encode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    return pd.get_dummies(df, columns=columns, drop_first=False)


def get_likert_columns(survey_def) -> List[str]:
    """Get all Likert scale column names from survey definition."""
    likert_qs = survey_def.get_questions_by_type("likert_scale")
    return [q.col_name for q in likert_qs]


def get_numeric_columns(survey_def) -> List[str]:
    """Get all numeric-like columns (continuous + likert) from survey definition."""
    cols = []
    for q in survey_def.questions:
        if q.var_type in ("continuous", "likert_scale", "nps"):
            cols.append(q.col_name)
    return cols


def get_categorical_columns(survey_def) -> List[str]:
    """Get all categorical columns (single_choice, multi_choice) from survey definition."""
    cols = []
    for q in survey_def.questions:
        if q.var_type in ("single_choice", "multi_choice", "demographic"):
            cols.append(q.col_name)
    return cols


def get_text_columns(survey_def) -> List[str]:
    """Get all text columns from survey definition."""
    cols = []
    for q in survey_def.questions:
        if q.var_type == "text":
            cols.append(q.col_name)
    return cols


def format_pvalue(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "p < 0.001***"
    elif p < 0.01:
        return f"p = {p:.4f}**"
    elif p < 0.05:
        return f"p = {p:.4f}*"
    else:
        return f"p = {p:.4f}"


def significance_stars(p: float) -> str:
    """Return significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"
