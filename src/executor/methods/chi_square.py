"""卡方检验 — 独立性检验 + Cramer's V."""

import pandas as pd
import numpy as np
from typing import Tuple
from scipy.stats import chi2_contingency

from ...utils.data_utils import get_categorical_columns, format_pvalue


def check_applicability(survey_def) -> Tuple[bool, str]:
    cat = get_categorical_columns(survey_def)
    if len(cat) < 2:
        return False, "需要至少2个分类变量"
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Run chi-square tests for all pairs of categorical variables."""
    df = survey_def.df
    cat_cols = get_categorical_columns(survey_def)
    tables = []
    warnings = []

    n_tested = 0

    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            col1, col2 = cat_cols[i], cat_cols[j]
            if col1 not in df.columns or col2 not in df.columns:
                continue

            # Build contingency table
            crosstab = pd.crosstab(df[col1], df[col2])
            if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
                continue
            if crosstab.values.min() < 1:
                warnings.append(f"{col1[:30]} × {col2[:30]}: 期望频数过小")
                continue

            try:
                chi2, p, dof, expected = chi2_contingency(crosstab)

                # Cramer's V
                n = crosstab.sum().sum()
                min_dim = min(crosstab.shape[0], crosstab.shape[1]) - 1
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

                n_tested += 1
                tables.append({
                    "title": f"卡方检验: {col1[:30]} × {col2[:30]}",
                    "data": {
                        "χ²": round(chi2, 2),
                        "自由度": dof,
                        "p值": format_pvalue(p),
                        "Cramer's V": round(cramers_v, 3),
                        "样本量": int(n),
                        "显著性": "显著" if p < 0.05 else "不显著",
                        "效应量解释": _interpret_cramers_v(cramers_v),
                    },
                })
            except Exception as e:
                warnings.append(f"卡方检验异常 ({col1[:20]} × {col2[:20]}): {e}")

    interpretation = (
        f"共进行 {n_tested} 组卡方独立性检验。  \n"
        f"p < 0.05 表示两个分类变量之间存在显著关联。  \n"
        f"Cramer's V 衡量关联强度: <0.1 弱, 0.1-0.3 中等, >0.3 强。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }


def _interpret_cramers_v(v: float) -> str:
    if v >= 0.5:
        return "强关联"
    elif v >= 0.3:
        return "中等偏强"
    elif v >= 0.1:
        return "中等"
    else:
        return "弱关联或无关联"
