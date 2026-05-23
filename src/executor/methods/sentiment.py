"""情感分析 — SnowNLP 中文情感分析."""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

from ...utils.data_utils import get_text_columns

logger = logging.getLogger(__name__)


def check_applicability(survey_def) -> Tuple[bool, str]:
    text_cols = get_text_columns(survey_def)
    if not text_cols:
        return False, "未检测到开放文本题"
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Run Chinese sentiment analysis using SnowNLP."""
    df = survey_def.df
    text_cols = get_text_columns(survey_def)
    tables = []
    warnings = []

    try:
        from snownlp import SnowNLP
    except ImportError:
        return {
            "tables": [],
            "charts": [],
            "interpretation": "SnowNLP 未安装，请运行: pip install snownlp",
            "warnings": ["SnowNLP 未安装"],
        }

    for col in text_cols:
        if col not in df.columns:
            continue

        texts = df[col].dropna().astype(str)
        texts = texts[texts.str.len() >= 5]  # Skip very short

        if len(texts) < 5:
            continue

        # Compute sentiment scores
        scores = []
        for text in texts:
            try:
                s = SnowNLP(text)
                scores.append(s.sentiments)
            except Exception:
                scores.append(0.5)

        scores = np.array(scores)
        positive = (scores > 0.6).sum()
        neutral = ((scores >= 0.4) & (scores <= 0.6)).sum()
        negative = (scores < 0.4).sum()
        total = len(scores)

        tables.append({
            "title": f"情感分析: {col[:60]}",
            "data": {
                "分析文本数": total,
                "正面 (>0.6)": f"{positive}条 ({positive/total*100:.1f}%)",
                "中性 (0.4-0.6)": f"{neutral}条 ({neutral/total*100:.1f}%)",
                "负面 (<0.4)": f"{negative}条 ({negative/total*100:.1f}%)",
                "平均情感得分": round(scores.mean(), 3),
                "情感倾向": "正面为主" if scores.mean() > 0.55 else (
                    "负面为主" if scores.mean() < 0.45 else "中性"
                ),
            },
        })

    if not tables:
        return {"tables": [], "charts": [], "interpretation": "无有效文本数据", "warnings": ["文本过短或数量不足"]}

    interpretation = (
        "SnowNLP 基于朴素贝叶斯模型对中文文本进行情感打分 [0, 1]。  \n"
        "得分 > 0.6 为正面，0.4-0.6 为中性，< 0.4 为负面。  \n"
        "注意：SnowNLP 基于电商评论训练，对特定领域文本可能需要校准。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }
