"""回归分析 — 线性/Logistic/Lasso 回归."""

import pandas as pd
import numpy as np
from typing import Tuple
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler

from ...utils.data_utils import get_numeric_columns, ensure_numeric, format_pvalue


def check_applicability(survey_def) -> Tuple[bool, str]:
    """Need at least 3 numeric columns (1 DV + 2+ IVs)."""
    num = get_numeric_columns(survey_def)
    if len(num) < 3:
        return False, f"需要至少3个连续/量表变量（1个因变量+2个自变量），当前 {len(num)} 个"
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Run linear regression on numeric variables.

    Uses the last Likert column as tentative DV.
    In practice, user should specify DV via the UI.
    """
    df = survey_def.df
    num_cols = get_numeric_columns(survey_def)[:15]
    data = ensure_numeric(df, num_cols).dropna()

    if len(data) < 30:
        return {"tables": [], "charts": [], "interpretation": "样本量不足（<30）", "warnings": ["样本量<30"]}

    # Use the first Likert column as tentative DV, rest as IVs
    likert_cols = survey_def.get_questions_by_type("likert_scale")
    if likert_cols:
        dv_col = likert_cols[0].col_name
    else:
        dv_col = num_cols[-1]

    iv_cols = [c for c in num_cols if c != dv_col and c in data.columns]
    if len(iv_cols) < 2:
        return {"tables": [], "charts": [], "interpretation": "自变量不足", "warnings": ["自变量<2"]}

    tables = []
    warnings = []

    # ── OLS Regression ──
    try:
        X = sm.add_constant(data[iv_cols].astype(float))
        y = data[dv_col].astype(float)
        model = sm.OLS(y, X).fit()

        # Coefficients table
        coef_rows = []
        for var in model.params.index:
            coef_rows.append({
                "变量": var[:50],
                "系数": round(model.params[var], 4),
                "标准误": round(model.bse[var], 4),
                "t值": round(model.tvalues[var], 3),
                "p值": format_pvalue(model.pvalues[var]),
                "显著性": "***" if model.pvalues[var] < 0.001 else (
                    "**" if model.pvalues[var] < 0.01 else (
                        "*" if model.pvalues[var] < 0.05 else ""
                    )
                ),
            })

        tables.append({
            "title": f"线性回归: {dv_col[:40]} 为因变量",
            "data": coef_rows,
        })

        # Model summary
        tables.append({
            "title": "回归模型摘要",
            "data": {
                "R²": round(model.rsquared, 3),
                "调整R²": round(model.rsquared_adj, 3),
                "F统计量": round(model.fvalue, 2),
                "F检验p值": format_pvalue(model.f_pvalue),
                "样本量": int(model.nobs),
                "因变量": dv_col[:40],
                "自变量数": len(iv_cols),
            },
        })

        # ── VIF (multicollinearity) ──
        try:
            vif_data = pd.DataFrame({
                "变量": iv_cols,
                "VIF": [round(variance_inflation_factor(X.values, i+1), 2) for i in range(len(iv_cols))],
            })
            high_vif = vif_data[vif_data["VIF"] > 10]
            if len(high_vif) > 0:
                warnings.append(f"{len(high_vif)} 个变量VIF>10，存在多重共线性")

            tables.append({
                "title": "多重共线性诊断 (VIF)",
                "data": vif_data.to_dict("records"),
            })
        except Exception:
            pass

    except Exception as e:
        warnings.append(f"回归分析异常: {e}")

    # ── Lasso (variable selection) ──
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data[iv_cols].astype(float))
        y_data = data[dv_col].astype(float)

        lasso = Lasso(alpha=0.1, max_iter=5000)
        lasso.fit(X_scaled, y_data)

        lasso_coefs = []
        for i, col in enumerate(iv_cols):
            lasso_coefs.append({
                "变量": col[:50],
                "Lasso系数": round(lasso.coef_[i], 4),
                "是否保留": "✓" if abs(lasso.coef_[i]) > 0.001 else "✗（被压缩为0）",
            })

        tables.append({
            "title": "Lasso 变量选择 (α=0.1)",
            "data": lasso_coefs,
        })
    except Exception as e:
        warnings.append(f"Lasso回归异常: {e}")

    interpretation = (
        f"以 '{dv_col[:40]}' 为因变量建立线性回归模型。  \n"
        f"R² = {model.rsquared:.3f}，模型解释了 {model.rsquared*100:.1f}% 的方差。  \n"
        f"显著的预测变量（p < 0.05）对因变量有统计学意义的影响。  \n"
        f"VIF > 10 的变量存在多重共线性，建议考虑删除或合并。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }
