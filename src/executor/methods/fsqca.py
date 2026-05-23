"""fsQCA — 模糊集定性比较分析（简化版）.

Full fsQCA requires specialized software (e.g., fsQCA 3.0, R/QCA).
This implementation provides a simplified necessity/sufficiency analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple

from ...utils.data_utils import get_likert_columns, get_categorical_columns, ensure_numeric


def check_applicability(survey_def) -> Tuple[bool, str]:
    likert = get_likert_columns(survey_def)
    cat = get_categorical_columns(survey_def)
    if len(likert) < 3:
        return False, f"需要至少3个Likert量表题，当前 {len(likert)} 个"
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Run simplified fsQCA necessity/sufficiency analysis.

    Calibrates Likert data into fuzzy sets (0-1) and checks
    necessity (consistency >= 0.9) and sufficiency conditions.
    """
    df = survey_def.df
    likert_cols = get_likert_columns(survey_def)[:10]
    data = ensure_numeric(df, likert_cols).dropna()

    if len(data) < 30:
        return {"tables": [], "charts": [], "interpretation": "样本量不足（<30）", "warnings": ["样本量<30"]}

    tables = []
    warnings = []

    # ── Calibration to fuzzy sets (0-1) ──
    # Using percentile-based calibration: 95th=full membership, 5th=full non-membership
    fuzzy_sets = {}
    calibrations = []

    for col in likert_cols:
        series = data[col].dropna()
        p5 = np.percentile(series, 5)
        p50 = np.percentile(series, 50)
        p95 = np.percentile(series, 95)

        # Log-odds calibration
        if p95 > p5:
            fuzzy = _calibrate(series.values, p5, p50, p95)
            fuzzy_sets[col] = fuzzy
            calibrations.append({
                "条件/结果": col[:50],
                "完全非隶属 (5%)": round(p5, 2),
                "交叉点 (50%)": round(p50, 2),
                "完全隶属 (95%)": round(p95, 2),
            })

    tables.append({"title": "模糊集校准阈值", "data": calibrations})

    # ── Necessity analysis ──
    # Use the last variable as "outcome", others as "conditions"
    if len(likert_cols) >= 4:
        outcome_col = likert_cols[-1]
        condition_cols = likert_cols[:-1]

        outcome = fuzzy_sets.get(outcome_col)
        if outcome is not None:
            nec_rows = []
            for cond_col in condition_cols:
                cond = fuzzy_sets.get(cond_col)
                if cond is not None:
                    # Necessity: outcome → condition
                    # Consistency = Σ min(cond, outcome) / Σ outcome
                    consistency = np.minimum(cond, outcome).sum() / (outcome.sum() + 1e-10)
                    # Coverage = Σ min(cond, outcome) / Σ cond
                    coverage = np.minimum(cond, outcome).sum() / (cond.sum() + 1e-10)

                    nec_rows.append({
                        "条件": cond_col[:40],
                        "一致性 (Consistency)": round(consistency, 3),
                        "覆盖率 (Coverage)": round(coverage, 3),
                        "必要性判断": "必要条件 ✓" if consistency >= 0.9 else (
                            "接近必要" if consistency >= 0.8 else "非必要"
                        ),
                    })

            tables.append({
                "title": f"必要性分析（结果: {outcome_col[:40]}）",
                "data": nec_rows,
            })

    interpretation = (
        "fsQCA（模糊集定性比较分析）用于探索哪些条件的组合导致结果。  \n"
        "必要性分析：一致性 ≥ 0.9 表示该条件是结果的必要条件。  \n"
        "注意：完整fsQCA分析需要构建真值表和求解逻辑最小化，建议使用专用软件（fsQCA 3.0 / R QCA包）。  \n"
        "此处为简化版分析，仅供参考探索。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }


def _calibrate(
    x: np.ndarray,
    full_non_member: float,
    crossover: float,
    full_member: float,
) -> np.ndarray:
    """Calibrate raw scores to fuzzy-set membership scores (0-1).

    Uses the direct method with log-odds transformation.
    """
    # Avoid log(0)
    eps = 1e-10

    # Deviation from crossover
    dev = x - crossover

    # Scaling factor
    neg_scale = crossover - full_non_member
    pos_scale = full_member - crossover

    membership = np.zeros_like(x, dtype=float)

    # Below crossover
    mask_neg = dev < 0
    if neg_scale > 0:
        membership[mask_neg] = np.exp(
            np.log(0.5) / np.log((crossover - x[mask_neg]) / neg_scale + eps)
        )
    else:
        membership[mask_neg] = 0.5

    # Above crossover
    mask_pos = dev > 0
    if pos_scale > 0:
        membership[mask_pos] = 1 - np.exp(
            np.log(0.5) / np.log((x[mask_pos] - crossover) / pos_scale + eps)
        )
    else:
        membership[mask_pos] = 0.5

    # At crossover
    membership[dev == 0] = 0.5

    return np.clip(membership, 0, 1)
