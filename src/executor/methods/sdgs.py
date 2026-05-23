"""SDGs 映射 — 将文本映射到17个可持续发展目标."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

from ...utils.data_utils import get_text_columns


# SDG keywords mapping (simplified)
SDG_KEYWORDS: Dict[str, List[str]] = {
    "SDG1-无贫穷": ["贫穷", "贫困", "脱贫", "温饱", "基本生活"],
    "SDG2-零饥饿": ["饥饿", "粮食", "食品安全", "营养", "农业"],
    "SDG3-良好健康与福祉": ["健康", "医疗", "卫生", "疾病", "疫苗", "心理"],
    "SDG4-优质教育": ["教育", "学习", "学校", "培训", "知识", "技能"],
    "SDG5-性别平等": ["性别", "女性", "平等", "妇女", "歧视"],
    "SDG6-清洁饮水和卫生": ["水", "饮水", "卫生", "厕所", "污水"],
    "SDG7-经济适用的清洁能源": ["能源", "电力", "清洁能源", "太阳能", "风能"],
    "SDG8-体面工作和经济增长": ["就业", "工作", "收入", "经济", "失业", "薪水"],
    "SDG9-产业创新和基础设施": ["创新", "技术", "基础设施", "工业", "数字化"],
    "SDG10-减少不平等": ["不平等", "公平", "差距", "贫富", "包容"],
    "SDG11-可持续城市和社区": ["城市", "交通", "住房", "社区", "公共空间"],
    "SDG12-负责任消费和生产": ["消费", "浪费", "回收", "循环", "可持续消费"],
    "SDG13-气候行动": ["气候", "碳排放", "温室", "全球变暖", "极端天气"],
    "SDG14-水下生物": ["海洋", "渔业", "塑料污染", "海洋生物", "珊瑚"],
    "SDG15-陆地生物": ["森林", "生物多样性", "生态系统", "物种", "沙漠化"],
    "SDG16-和平正义与强大机构": ["治理", "法治", "腐败", "透明", "政府"],
    "SDG17-促进目标实现的伙伴关系": ["合作", "伙伴", "国际", "援助", "发展"],
}


def check_applicability(survey_def) -> Tuple[bool, str]:
    text_cols = get_text_columns(survey_def)
    if not text_cols:
        return False, "未检测到开放文本题"
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Map text responses to SDGs via keyword matching."""
    df = survey_def.df
    text_cols = get_text_columns(survey_def)
    tables = []
    warnings = []

    for col in text_cols:
        if col not in df.columns:
            continue

        texts = df[col].dropna().astype(str)
        if len(texts) < 3:
            continue

        # Count keyword hits per SDG
        sdg_counts = {goal: 0 for goal in SDG_KEYWORDS}

        for text in texts:
            text_lower = text.lower()
            for goal, keywords in SDG_KEYWORDS.items():
                for kw in keywords:
                    if kw in text_lower:
                        sdg_counts[goal] += 1
                        break  # Count each SDG at most once per text

        # Sort by count
        sorted_sdgs = sorted(sdg_counts.items(), key=lambda x: x[1], reverse=True)
        total_texts = len(texts)

        sdg_rows = []
        for goal, count in sorted_sdgs:
            if count > 0:
                sdg_rows.append({
                    "SDG目标": goal,
                    "提及次数": count,
                    "覆盖率": f"{count/total_texts*100:.1f}%",
                })

        if sdg_rows:
            tables.append({
                "title": f"SDGs映射: {col[:50]}",
                "data": sdg_rows,
            })

    if not tables:
        return {"tables": [], "charts": [], "interpretation": "文本未匹配到SDGs关键词", "warnings": warnings}

    interpretation = (
        "基于17个可持续发展目标（SDGs）的关键词进行文本映射。  \n"
        "覆盖率表示提及该SDG相关议题的文本占比。  \n"
        "注意：这是一种粗略的文本主题标注方法，仅供初步探索。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }
