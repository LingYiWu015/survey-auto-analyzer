"""描述统计与列联表 — 频数/百分比/均值±标准差."""

import pandas as pd
import numpy as np
from typing import Tuple

from ...utils.data_utils import (
    get_numeric_columns, get_categorical_columns,
    ensure_numeric, format_pvalue,
)


def check_applicability(survey_def) -> Tuple[bool, str]:
    """Always applicable."""
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Run descriptive statistics and crosstabs."""
    df = survey_def.df
    tables = []
    warnings = []

    # ── Sample overview ──
    tables.append({
        "title": "样本概况",
        "data": {
            "总样本量": len(df),
            "题目总数": survey_def.n_questions,
            "数据来源": survey_def.platform,
        },
    })

    # ── Numeric columns: mean ± std ──
    numeric_cols = get_numeric_columns(survey_def)
    if numeric_cols:
        num_df = ensure_numeric(df, numeric_cols)[numeric_cols]
        stats = num_df.describe().T
        stats["缺失数"] = df[numeric_cols].isna().sum().values
        stats["缺失率"] = (df[numeric_cols].isna().mean() * 100).round(1).values

        summary = []
        for col in numeric_cols:
            if col in stats.index:
                s = stats.loc[col]
                summary.append({
                    "题目": col[:50],
                    "均值": round(s["mean"], 2),
                    "标准差": round(s["std"], 2),
                    "最小值": s["min"],
                    "最大值": s["max"],
                    "缺失率": f"{s['缺失率']}%",
                })

        tables.append({
            "title": "连续变量描述统计",
            "data": summary,
        })

    # ── Categorical columns: frequency ──
    cat_cols = get_categorical_columns(survey_def)
    if cat_cols:
        for col in cat_cols[:10]:  # Limit to 10 crosstabs
            if col not in df.columns:
                continue
            freq = df[col].value_counts().head(15)  # Top 15 categories
            pct = (df[col].value_counts(normalize=True) * 100).round(1).head(15)

            cat_summary = []
            for val in freq.index:
                cat_summary.append({
                    "选项": str(val),
                    "频数": int(freq[val]),
                    "百分比": f"{pct[val]:.1f}%",
                })

            tables.append({
                "title": f"频数分布: {col[:60]}",
                "data": cat_summary,
            })

    # ── Demographics summary ──
    demo_cols = survey_def.get_questions_by_type("demographic")
    if demo_cols:
        demo_text = "**样本人口统计特征：**  \n"
        for q in demo_cols[:5]:
            if q.col_name in df.columns:
                top_val = df[q.col_name].value_counts().index[0]
                top_pct = df[q.col_name].value_counts(normalize=True).iloc[0] * 100
                demo_text += f"- {q.label}: {top_val}占比最高（{top_pct:.1f}%）  \n"
    else:
        demo_text = ""

    interpretation = (
        f"本次调查共收集 {len(df)} 份有效样本。  \n"
        f"{demo_text}"
        f"共包含 {len(numeric_cols)} 个连续/量表变量和 {len(cat_cols)} 个分类变量。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }
