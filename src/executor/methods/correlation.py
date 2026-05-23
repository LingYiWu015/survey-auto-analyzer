"""相关分析 — Pearson / Spearman 相关系数矩阵."""

import pandas as pd
import numpy as np
from typing import Tuple
from scipy import stats

from ...utils.data_utils import get_numeric_columns, ensure_numeric


def check_applicability(survey_def) -> Tuple[bool, str]:
    """Need at least 2 numeric columns."""
    num = get_numeric_columns(survey_def)
    if len(num) < 2:
        return False, f"需要至少2个连续/量表变量，当前 {len(num)} 个"
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Compute Pearson and Spearman correlation matrices."""
    df = survey_def.df
    num_cols = get_numeric_columns(survey_def)[:20]  # Cap at 20
    data = ensure_numeric(df, num_cols)[num_cols].dropna()

    if len(data) < 10:
        return {"tables": [], "charts": [], "interpretation": "样本量不足", "warnings": ["有效样本<10"]}

    tables = []
    warnings = []

    # ── Pearson ──
    try:
        pearson = data.corr(method="pearson")
        tables.append({
            "title": "Pearson 相关系数矩阵",
            "data": _format_corr_matrix(pearson),
        })
    except Exception as e:
        warnings.append(f"Pearson相关计算异常: {e}")

    # ── Spearman ──
    try:
        spearman = data.corr(method="spearman")
        tables.append({
            "title": "Spearman 相关系数矩阵",
            "data": _format_corr_matrix(spearman),
        })
    except Exception as e:
        warnings.append(f"Spearman相关计算异常: {e}")

    # ── Top correlations ──
    try:
        top_pairs = _top_correlations(pearson if 'pearson' in dir() else data.corr(method="pearson"))
        tables.append({
            "title": "最强相关关系 Top 10",
            "data": top_pairs,
        })
    except Exception:
        pass

    interpretation = (
        f"分析了 {len(num_cols)} 个连续/量表变量之间的相关关系。  \n"
        f"Pearson 相关系数衡量线性相关程度，r > 0 为正相关，r < 0 为负相关。  \n"
        f"|r| > 0.7 为强相关，0.4-0.7 为中等相关，< 0.4 为弱相关。"
    )

    return {
        "tables": tables,
        "charts": _build_chart_data(pearson if 'pearson' in dir() else data.corr()),
        "interpretation": interpretation,
        "warnings": warnings,
    }


def _format_corr_matrix(corr_df: pd.DataFrame) -> list:
    """Format correlation matrix as list of dicts."""
    rows = []
    cols = corr_df.columns
    for i, col_i in enumerate(cols):
        row_data = {"变量": col_i[:50]}
        for col_j in cols:
            if col_i != col_j:
                short_j = col_j[:30]
                row_data[short_j] = round(corr_df.loc[col_i, col_j], 3)
        rows.append(row_data)
    return rows


def _top_correlations(corr_df: pd.DataFrame, n: int = 10) -> list:
    """Extract top N strongest absolute correlations."""
    pairs = []
    cols = corr_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({
                "变量A": cols[i][:40],
                "变量B": cols[j][:40],
                "相关系数": round(corr_df.iloc[i, j], 3),
                "绝对值": abs(corr_df.iloc[i, j]),
            })
    pairs.sort(key=lambda x: x["绝对值"], reverse=True)
    return pairs[:n]


def _build_chart_data(corr_df: pd.DataFrame) -> list:
    """Build chart data for heatmap."""
    return [{
        "type": "heatmap",
        "title": "相关系数热力图",
        "data": corr_df.round(3).to_dict(),
        "labels": list(corr_df.columns),
    }]
