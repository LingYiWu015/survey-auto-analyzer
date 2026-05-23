"""CCSI / ACSI 满意度指数模型 — SEM结构方程模型."""

import pandas as pd
import numpy as np
from typing import Tuple

from ...utils.data_utils import get_likert_columns, ensure_numeric


def check_applicability(survey_def) -> Tuple[bool, str]:
    likert = get_likert_columns(survey_def)
    if len(likert) < 5:
        return False, f"需要至少5个Likert量表题构建SEM，当前 {len(likert)} 个"
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Run simplified CCSI/ACSI analysis using semopy.

    CCSI (China Customer Satisfaction Index) model:
    Brand Image → Perceived Quality → Perceived Value → Satisfaction → Loyalty
    """
    df = survey_def.df
    likert_cols = get_likert_columns(survey_def)[:15]
    data = ensure_numeric(df, likert_cols).dropna()

    if len(data) < 100:
        return {
            "tables": [], "charts": [],
            "interpretation": "样本量不足（建议≥100），SEM模型需要较大样本量",
            "warnings": ["样本量<100，SEM估计可能不稳定"],
        }

    tables = []
    warnings = []

    try:
        import semopy
    except ImportError:
        return {
            "tables": [], "charts": [],
            "interpretation": "semopy 未安装，请运行: pip install semopy",
            "warnings": ["semopy 未安装"],
        }

    try:
        # Build a simplified CCSI model
        # Rename columns for model specification
        col_map = {c: f"X{i+1}" for i, c in enumerate(likert_cols)}
        model_data = data.rename(columns=col_map)
        var_names = list(col_map.values())
        n_vars = len(var_names)

        # Simple measurement model: all variables load on one latent factor "Satisfaction"
        model_spec = "# 简化满意度测量模型\n"
        model_spec += "Satisfaction =~ " + " + ".join(var_names) + "\n"

        model = semopy.Model(model_spec)
        model.fit(model_data)

        # Get parameter estimates
        estimates = model.inspect()
        param_rows = []
        for _, row in estimates.iterrows():
            param_rows.append({
                "参数": str(row.get("lval", "")) + " " + str(row.get("op", "")) + " " + str(row.get("rval", "")),
                "估计值": round(row.get("Estimate", 0), 3),
                "标准误": round(row.get("Std. Err", 0), 3),
                "z值": round(row.get("z-value", 0), 2),
                "p值": f"{row.get('p-value', 1):.4f}",
            })

        tables.append({
            "title": "SEM 参数估计（简化CCSI模型）",
            "data": param_rows,
        })

        # Fit indices
        stats = semopy.calc_stats(model)
        fit_rows = []
        for stat_name in ["chi2", "df", "p-value", "GFI", "AGFI", "CFI", "RMSEA", "SRMR"]:
            try:
                val = getattr(stats, stat_name, None)
                if val is not None:
                    fit_rows.append({"指标": stat_name, "值": round(float(val), 3)})
            except Exception:
                pass

        if fit_rows:
            tables.append({"title": "模型拟合指标", "data": fit_rows})

    except Exception as e:
        warnings.append(f"SEM建模异常: {e}")

    interpretation = (
        "CCSI（中国顾客满意度指数）模型通过结构方程模型（SEM）估计满意度及其驱动因素。  \n"
        "由于未预设模型结构，此处使用简化模型（单因子测量模型）。  \n"
        "完整CCSI模型包含品牌形象、感知质量、感知价值、满意度和忠诚度五个构念。  \n"
        "建议：在Streamlit界面中手动指定各题目所属维度以构建完整模型。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }
