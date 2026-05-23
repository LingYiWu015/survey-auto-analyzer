"""NPS 净推荐值 — 推荐者/被动者/贬损者分组."""

import pandas as pd
import numpy as np
from typing import Tuple

from ...utils.data_utils import get_numeric_columns, ensure_numeric


def check_applicability(survey_def) -> Tuple[bool, str]:
    nps_qs = survey_def.get_questions_by_type("nps")
    if not nps_qs:
        return False, "未检测到NPS评分题（0-10分制）"
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Calculate NPS score and breakdown."""
    df = survey_def.df
    nps_qs = survey_def.get_questions_by_type("nps")
    tables = []
    warnings = []

    for q in nps_qs:
        col = q.col_name
        if col not in df.columns:
            continue

        scores = ensure_numeric(df, [col])[col].dropna()
        if len(scores) < 10:
            continue

        # NPS classification: 0-6 Detractor, 7-8 Passive, 9-10 Promoter
        detractors = (scores <= 6).sum()
        passives = ((scores >= 7) & (scores <= 8)).sum()
        promoters = (scores >= 9).sum()
        total = len(scores)

        nps = round((promoters - detractors) / total * 100, 1)

        tables.append({
            "title": f"NPS分析: {q.label[:50]}",
            "data": {
                "NPS得分": nps,
                "推荐者 (9-10分)": f"{promoters}人 ({promoters/total*100:.1f}%)",
                "被动者 (7-8分)": f"{passives}人 ({passives/total*100:.1f}%)",
                "贬损者 (0-6分)": f"{detractors}人 ({detractors/total*100:.1f}%)",
                "总样本": total,
                "均值": round(scores.mean(), 2),
                "标准差": round(scores.std(), 2),
            },
        })

        # NPS level interpretation
        if nps >= 70:
            level = "卓越（世界级）"
        elif nps >= 50:
            level = "优秀"
        elif nps >= 30:
            level = "良好"
        elif nps >= 0:
            level = "一般（有提升空间）"
        else:
            level = "较差（需重点关注）"

        tables.append({
            "title": f"NPS等级评价",
            "data": {"NPS": nps, "等级": level},
        })

    interpretation = (
        f"NPS（净推荐值）= 推荐者占比 - 贬损者占比，范围 [-100, 100]。  \n"
        f"正值表示推荐者多于贬损者，一般认为 NPS > 30 为良好，> 50 为优秀。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }
