"""游程检验 — 随机性检验（Runs Test）."""

import pandas as pd
import numpy as np
from typing import Tuple
from statsmodels.sandbox.stats.runs import runstest_1samp

from ...utils.data_utils import ensure_numeric


def check_applicability(survey_def) -> Tuple[bool, str]:
    """Only applicable for ordered sequence data."""
    if survey_def.sample_size < 30:
        return False, "样本量不足（<30）"
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Run Wald-Wolfowitz runs test for randomness on numeric columns."""
    df = survey_def.df
    tables = []
    warnings = []

    # Get numeric columns
    num_cols = []
    for q in survey_def.questions:
        if q.var_type in ("continuous", "likert_scale", "nps"):
            num_cols.append(q.col_name)

    if not num_cols:
        return {"tables": [], "charts": [], "interpretation": "无连续变量", "warnings": ["无连续变量"]}

    for col in num_cols[:5]:  # Limit to 5
        if col not in df.columns:
            continue

        series = ensure_numeric(df, [col])[col].dropna()

        if len(series) < 30:
            continue

        # Dichotomize at median
        median = series.median()
        binary = (series > median).astype(int).values

        try:
            z_stat, p_value = runstest_1samp(binary, correction=True)
            n_runs = _count_runs(binary)

            tables.append({
                "title": f"游程检验: {col[:50]}",
                "data": {
                    "变量": col[:50],
                    "中位数": round(median, 2),
                    "游程数": n_runs,
                    "Z统计量": round(z_stat, 3) if not np.isnan(z_stat) else "N/A",
                    "p值": f"{p_value:.4f}" if not np.isnan(p_value) else "N/A",
                    "结论": "随机序列" if p_value > 0.05 else "非随机序列（存在模式）",
                },
            })
        except Exception as e:
            warnings.append(f"游程检验异常 ({col[:30]}): {e}")

    interpretation = (
        "游程检验判断数据序列是否为随机排列。  \n"
        "p > 0.05 表示数据为随机序列（无明显模式），p < 0.05 表示存在非随机模式。  \n"
        "通常在问卷数据中，问题顺序固定，游程检验的实际应用价值有限。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }


def _count_runs(binary: np.ndarray) -> int:
    """Count the number of runs in a binary sequence."""
    if len(binary) < 2:
        return len(binary)
    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i - 1]:
            runs += 1
    return runs
