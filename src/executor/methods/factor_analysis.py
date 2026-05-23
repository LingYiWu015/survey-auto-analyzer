"""因子分析 — 探索性因子分析，主成分法，Varimax旋转."""

import pandas as pd
import numpy as np
from typing import Tuple
from factor_analyzer import FactorAnalyzer

from ...utils.data_utils import get_likert_columns, ensure_numeric


def check_applicability(survey_def) -> Tuple[bool, str]:
    likert = get_likert_columns(survey_def)
    if len(likert) < 4:
        return False, f"需要至少4个Likert量表题，当前 {len(likert)} 个"
    return True, ""


def run(survey_def, n_factors: int = None, **kwargs) -> dict:
    """Run exploratory factor analysis."""
    df = survey_def.df
    likert_cols = get_likert_columns(survey_def)
    data = ensure_numeric(df, likert_cols)[likert_cols].dropna()

    if len(data) < 50:
        return {"tables": [], "charts": [], "interpretation": "样本量不足（<50）", "warnings": ["样本量<50，因子分析不稳定"]}

    tables = []
    warnings = []

    # ── Determine number of factors ──
    try:
        fa_full = FactorAnalyzer(rotation=None, n_factors=len(likert_cols))
        fa_full.fit(data)
        eigenvalues, _ = fa_full.get_eigenvalues()

        # Kaiser criterion: eigenvalues > 1
        n_factors_kaiser = sum(1 for e in eigenvalues if e > 1)
        if n_factors is None:
            n_factors = max(1, n_factors_kaiser)
        n_factors = min(n_factors, len(likert_cols) - 1)

        tables.append({
            "title": "特征值与方差解释",
            "data": [
                {
                    "因子": i + 1,
                    "特征值": round(eigenvalues[i], 3),
                    "方差%": f"{round(eigenvalues[i] / len(likert_cols) * 100, 1)}%",
                    "累计%": f"{round(sum(eigenvalues[:i+1]) / len(likert_cols) * 100, 1)}%",
                    "Kaiser标准": ">1 ✓" if eigenvalues[i] > 1 else "<1",
                }
                for i in range(min(len(eigenvalues), 15))
            ],
        })
    except Exception as e:
        warnings.append(f"特征值计算异常: {e}")
        n_factors = 3

    # ── Factor Analysis with Varimax ──
    try:
        fa = FactorAnalyzer(rotation="varimax", n_factors=n_factors)
        fa.fit(data)

        # Factor loadings matrix
        loadings = pd.DataFrame(
            fa.loadings_,
            index=likert_cols,
            columns=[f"因子{i+1}" for i in range(n_factors)],
        )

        # Format as list
        loading_rows = []
        for col_name in loadings.index:
            row = {"题目": col_name[:50]}
            for fcol in loadings.columns:
                row[fcol] = round(loadings.loc[col_name, fcol], 3)
            # Find dominant factor
            abs_loads = [abs(loadings.loc[col_name, f"因子{i+1}"]) for i in range(n_factors)]
            dominant = abs_loads.index(max(abs_loads)) + 1
            row["归属因子"] = f"因子{dominant}"
            loading_rows.append(row)

        tables.append({
            "title": f"因子载荷矩阵 (Varimax旋转, {n_factors}因子)",
            "data": loading_rows,
        })

        # Variance explained per factor
        var_explained = []
        for i in range(n_factors):
            var_explained.append({
                "因子": f"因子{i+1}",
                "方差解释量": round(fa.get_factor_variance()[0][i], 3),
                "方差%": f"{round(fa.get_factor_variance()[1][i] * 100, 1)}%",
                "累计%": f"{round(fa.get_factor_variance()[2][i] * 100, 1)}%",
            })

        tables.append({
            "title": "旋转后方差解释",
            "data": var_explained,
        })

    except Exception as e:
        warnings.append(f"因子分析异常: {e}")

    total_var = sum(eigenvalues[:n_factors]) / len(likert_cols) * 100 if 'eigenvalues' in dir() else 0
    interpretation = (
        f"提取了 {n_factors} 个因子，累计解释 {total_var:.1f}% 的方差。  \n"
        f"每个题目的归属因子为绝对值载荷最大的因子。  \n"
        f"通常要求因子载荷 > 0.5 表示该题目较好地测量了对应因子。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }
