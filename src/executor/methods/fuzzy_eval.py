"""模糊综合评判 — 多维度综合评价."""

import pandas as pd
import numpy as np
from typing import Tuple

from ...utils.data_utils import get_likert_columns, ensure_numeric


def check_applicability(survey_def) -> Tuple[bool, str]:
    likert = get_likert_columns(survey_def)
    if len(likert) < 4:
        return False, f"需要至少4个Likert量表题，当前 {len(likert)} 个"
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Run fuzzy comprehensive evaluation."""
    df = survey_def.df
    likert_cols = get_likert_columns(survey_def)
    data = ensure_numeric(df, likert_cols)[likert_cols].dropna()

    if len(data) < 20:
        return {"tables": [], "charts": [], "interpretation": "样本量不足（<20）", "warnings": ["样本量<20"]}

    tables = []
    warnings = []

    # ── Equal weight fuzzy evaluation ──
    n_vars = len(likert_cols)
    weights = np.ones(n_vars) / n_vars  # Equal weights as default

    # Normalize to 0-1 scale
    data_min = data.min()
    data_max = data.max()
    range_val = data_max - data_min
    range_val = range_val.replace(0, 1)  # Avoid division by zero
    normalized = (data - data_min) / range_val

    # Weighted score
    scores = normalized.dot(weights)
    mean_score = scores.mean()
    std_score = scores.std()

    # Rating levels
    levels = ["很低", "较低", "中等", "较高", "很高"]
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    level_counts = []
    for i in range(len(levels)):
        count = ((scores >= bins[i]) & (scores < bins[i+1])).sum()
        level_counts.append({
            "评价等级": levels[i],
            "人数": int(count),
            "占比": f"{count/len(scores)*100:.1f}%",
        })

    tables.append({
        "title": "模糊综合评判结果（等权重）",
        "data": level_counts,
    })

    # Summary
    tables.append({
        "title": "综合得分摘要",
        "data": {
            "平均综合得分": round(mean_score, 3),
            "标准差": round(std_score, 3),
            "最低分": round(scores.min(), 3),
            "最高分": round(scores.max(), 3),
            "评价指标数": n_vars,
            "有效样本": len(data),
        },
    })

    interpretation = (
        f"基于 {n_vars} 个指标的模糊综合评判（等权重）。  \n"
        f"平均综合得分为 {mean_score:.3f}（0-1标度）。  \n"
        f"{level_counts[3]['占比']} 的受访者处于'较高'及以上水平。  \n"
        f"如需更精确的权重，建议使用AHP层次分析法或熵权法确定指标权重。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }
