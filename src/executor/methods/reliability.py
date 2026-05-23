"""信效度检验 — Cronbach's α + KMO + Bartlett."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

from ...utils.data_utils import ensure_numeric, get_likert_columns


def check_applicability(survey_def) -> Tuple[bool, str]:
    """Check if reliability analysis is applicable."""
    likert_cols = get_likert_columns(survey_def)
    if len(likert_cols) < 3:
        return False, f"需要至少3个Likert量表题，当前检测到 {len(likert_cols)} 个"
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Run reliability and validity tests."""
    from pingouin import cronbach_alpha
    from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

    df = survey_def.df
    likert_cols = get_likert_columns(survey_def)
    data = ensure_numeric(df, likert_cols)[likert_cols].dropna()

    if len(data) < 30:
        return {
            "tables": [],
            "charts": [],
            "interpretation": "样本量不足（<30），无法进行信效度检验",
            "warnings": ["样本量小于30，结果可能不稳定"],
        }

    tables = []
    warnings = []
    alpha_val = None
    kmo_val = None

    # ── Cronbach's α ──
    try:
        alpha, ci = cronbach_alpha(data)
        alpha_val = round(alpha, 3)
        ci_low, ci_high = round(ci[0], 3), round(ci[1], 3)

        if alpha_val >= 0.9:
            level = "优秀"
        elif alpha_val >= 0.8:
            level = "良好"
        elif alpha_val >= 0.7:
            level = "可接受"
        else:
            level = "偏低"
            warnings.append(f"Cronbach's α = {alpha_val}，低于0.7，信度{level}")

        tables.append({
            "title": "Cronbach's α 信度检验",
            "data": {
                "Cronbach's α": alpha_val,
                "95% CI 下限": ci_low,
                "95% CI 上限": ci_high,
                "评价": level,
                "题目数": len(likert_cols),
                "有效样本": len(data),
            },
        })
    except Exception as e:
        warnings.append(f"信度检验异常: {e}")

    # ── KMO & Bartlett ──
    try:
        kmo_result = calculate_kmo(data)
        if isinstance(kmo_result, (tuple, list)):
            kmo_val = float(kmo_result[0])
        else:
            kmo_val = float(kmo_result)

        chi2, p_value = calculate_bartlett_sphericity(data)

        if kmo_val >= 0.9:
            kmo_level = "极佳"
        elif kmo_val >= 0.8:
            kmo_level = "良好"
        elif kmo_val >= 0.7:
            kmo_level = "适中"
        elif kmo_val >= 0.6:
            kmo_level = "勉强可接受"
        else:
            kmo_level = "不适合因子分析"
            warnings.append(f"KMO = {kmo_val:.3f}，{kmo_level}")

        tables.append({
            "title": "KMO 与 Bartlett 球形检验",
            "data": {
                "KMO 值": round(kmo_val, 3),
                "KMO 评价": kmo_level,
                "Bartlett χ²": round(float(chi2), 2),
                "Bartlett p值": f"{float(p_value):.4f}",
                "Bartlett 显著性": "显著" if float(p_value) < 0.05 else "不显著",
            },
        })
    except Exception as e:
        warnings.append(f"效度检验异常: {e}")

    # ── Interpretation ──
    interpretation = _build_interpretation(alpha_val, kmo_val)

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }


def _build_interpretation(alpha: Optional[float], kmo: Optional[float]) -> str:
    """Build human-readable interpretation."""
    parts = []
    if alpha is not None:
        if alpha >= 0.8:
            parts.append(f"量表信度良好（Cronbach's α = {alpha:.3f}），内部一致性高。")
        elif alpha >= 0.7:
            parts.append(f"量表信度可接受（Cronbach's α = {alpha:.3f}），满足基本要求。")
        else:
            parts.append(f"量表信度偏低（Cronbach's α = {alpha:.3f}），建议检查或删除部分题目。")
    if kmo is not None:
        if kmo >= 0.7:
            parts.append(f"KMO = {kmo:.3f}，数据适合进行因子分析。")
        elif kmo >= 0.6:
            parts.append(f"KMO = {kmo:.3f}，数据勉强适合因子分析，建议增加样本量。")
        else:
            parts.append(f"KMO = {kmo:.3f}，数据不太适合因子分析，需考虑增加题目或样本。")
    return " ".join(parts)
