"""T检验与方差分析 — 独立样本T检验 + 单因素ANOVA + 事后比较."""

import pandas as pd
import numpy as np
from typing import Tuple
from scipy import stats
from pingouin import pairwise_tukey

from ...utils.data_utils import (
    get_numeric_columns, get_categorical_columns,
    ensure_numeric, format_pvalue, significance_stars,
)


def check_applicability(survey_def) -> Tuple[bool, str]:
    """Check: need categorical (grouping) + numeric (DV)."""
    num = get_numeric_columns(survey_def)
    cat = get_categorical_columns(survey_def)
    if not num:
        return False, "没有连续/量表变量"
    if not cat:
        return False, "没有分类变量作为分组依据"
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Run T-tests and ANOVA for all categorical × numeric pairs."""
    df = survey_def.df
    num_cols = get_numeric_columns(survey_def)
    cat_cols = get_categorical_columns(survey_def)
    tables = []
    warnings = []

    for cat_col in cat_cols:
        if cat_col not in df.columns:
            continue

        groups = df[cat_col].dropna().unique()
        n_groups = len(groups)

        if n_groups < 2:
            continue
        if n_groups > 20:
            continue  # Too many groups

        for num_col in num_cols:
            if num_col not in df.columns:
                continue

            # Prepare data
            valid = df[[cat_col, num_col]].dropna()
            num_data = ensure_numeric(valid, [num_col])

            if n_groups == 2:
                # ── T-test ──
                g1 = num_data[num_data[cat_col] == groups[0]][num_col].dropna()
                g2 = num_data[num_data[cat_col] == groups[1]][num_col].dropna()

                if len(g1) < 3 or len(g2) < 3:
                    continue

                t_stat, p_val = stats.ttest_ind(g1, g2)
                m1, m2 = g1.mean(), g2.mean()

                tables.append({
                    "title": f"T检验: {num_col[:40]} × {cat_col[:30]}",
                    "data": {
                        "分组变量": cat_col[:40],
                        "因变量": num_col[:40],
                        f"{groups[0]} (均值)": round(m1, 2),
                        f"{groups[1]} (均值)": round(m2, 2),
                        "均值差": round(m1 - m2, 2),
                        "t 统计量": round(t_stat, 3),
                        "p 值": format_pvalue(p_val),
                        "显著性": significance_stars(p_val),
                        f"{groups[0]} 样本量": len(g1),
                        f"{groups[1]} 样本量": len(g2),
                    },
                })
            else:
                # ── ANOVA ──
                group_data = []
                for g in groups:
                    vals = num_data[num_data[cat_col] == g][num_col].dropna()
                    if len(vals) >= 3:
                        group_data.append(vals)

                if len(group_data) < 3:
                    continue

                try:
                    f_stat, p_val = stats.f_oneway(*group_data)

                    tables.append({
                        "title": f"ANOVA: {num_col[:40]} × {cat_col[:30]}",
                        "data": {
                            "分组变量": cat_col[:40],
                            "因变量": num_col[:40],
                            "组数": n_groups,
                            "F 统计量": round(f_stat, 3),
                            "p 值": format_pvalue(p_val),
                            "显著性": significance_stars(p_val),
                        },
                    })

                    # Post-hoc Tukey if significant
                    if p_val < 0.05:
                        try:
                            tukey = pairwise_tukey(
                                dv=num_col, between=cat_col, data=num_data
                            )
                            sig_pairs = tukey[tukey["p-tukey"] < 0.05]
                            if len(sig_pairs) > 0:
                                posthoc_rows = []
                                for _, row in sig_pairs.head(10).iterrows():
                                    posthoc_rows.append({
                                        "组A": str(row["A"]),
                                        "组B": str(row["B"]),
                                        "均值差": round(row["diff"], 2),
                                        "p值": format_pvalue(row["p-tukey"]),
                                    })
                                tables.append({
                                    "title": f"事后比较 (Tukey HSD): {num_col[:40]}",
                                    "data": posthoc_rows,
                                })
                        except Exception:
                            pass

                except Exception as e:
                    warnings.append(f"ANOVA异常 ({cat_col} × {num_col}): {e}")

    n_tests = len([t for t in tables if "T检验" in t.get("title", "") or "ANOVA" in t.get("title", "")])
    interpretation = (
        f"共进行了 {n_tests} 组差异比较分析（T检验/ANOVA）。  \n"
        f"当 p < 0.05 时，表示组间存在显著差异，标记为 *；"
        f"p < 0.01 为 **；p < 0.001 为 ***。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }
