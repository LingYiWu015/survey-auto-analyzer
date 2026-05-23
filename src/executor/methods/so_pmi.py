"""SO-PMI 情感词典构建 — 点互信息法."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Set
from collections import Counter

from ...utils.data_utils import get_text_columns


# Seed words for sentiment orientation
POSITIVE_SEEDS = {"好", "优秀", "满意", "喜欢", "棒", "赞", "方便", "实惠", "舒适", "专业"}
NEGATIVE_SEEDS = {"差", "糟糕", "失望", "讨厌", "垃圾", "麻烦", "贵", "难受", "不足", "缺陷"}


def check_applicability(survey_def) -> Tuple[bool, str]:
    text_cols = get_text_columns(survey_def)
    if len(text_cols) < 3:
        return False, f"建议至少3个文本题构建词典更有意义，当前 {len(text_cols)} 个"
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Build domain sentiment dictionary using SO-PMI."""
    df = survey_def.df
    text_cols = get_text_columns(survey_def)
    tables = []
    warnings = []

    try:
        import jieba
    except ImportError:
        return {"tables": [], "charts": [], "interpretation": "jieba 未安装", "warnings": ["jieba 未安装"]}

    # Collect all texts
    all_words = []
    for col in text_cols:
        if col in df.columns:
            texts = df[col].dropna().astype(str)
            for text in texts:
                words = jieba.lcut(text)
                words = [w.strip() for w in words if len(w.strip()) >= 2]
                all_words.extend(words)

    if len(all_words) < 100:
        return {"tables": [], "charts": [], "interpretation": "文本量不足以构建词典", "warnings": ["词数<100"]}

    total = len(all_words)
    word_counts = Counter(all_words)

    # SO-PMI for each word with frequency >= 3
    candidate_words = {w for w, c in word_counts.items() if c >= 3 and w not in POSITIVE_SEEDS and w not in NEGATIVE_SEEDS}

    so_pmi_scores = {}
    for word in candidate_words:
        p_word = word_counts[word] / total

        pos_pmi = 0
        for pw in POSITIVE_SEEDS:
            co_count = _co_occurrence_count(all_words, word, pw)
            if co_count > 0:
                p_co = co_count / (total - 1)
                p_pos = word_counts.get(pw, 0) / total
                if p_co > 0 and p_word > 0 and p_pos > 0:
                    pos_pmi += np.log2(p_co / (p_word * p_pos))

        neg_pmi = 0
        for nw in NEGATIVE_SEEDS:
            co_count = _co_occurrence_count(all_words, word, nw)
            if co_count > 0:
                p_co = co_count / (total - 1)
                p_neg = word_counts.get(nw, 0) / total
                if p_co > 0 and p_word > 0 and p_neg > 0:
                    neg_pmi += np.log2(p_co / (p_word * p_neg))

        so_pmi_scores[word] = round(pos_pmi - neg_pmi, 4)

    # Top positive and negative words
    sorted_words = sorted(so_pmi_scores.items(), key=lambda x: x[1], reverse=True)

    top_pos = [{"词语": w, "SO-PMI": s} for w, s in sorted_words[:20] if s > 0]
    top_neg = [{"词语": w, "SO-PMI": s} for w, s in sorted_words[-20:] if s < 0]

    if top_pos:
        tables.append({"title": "正向情感词 Top 20", "data": top_pos})
    if top_neg:
        tables.append({"title": "负向情感词 Top 20", "data": top_neg})

    interpretation = (
        "SO-PMI（情感倾向点互信息）基于词语与情感种子词的共现计算情感倾向。  \n"
        "SO-PMI > 0 表示词语倾向于正向语境，< 0 倾向于负向语境。  \n"
        f"共分析了 {len(so_pmi_scores)} 个候选词的情感倾向。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }


def _co_occurrence_count(words: List[str], word: str, seed: str, window: int = 5) -> int:
    """Count co-occurrences within a sliding window."""
    count = 0
    for i, w in enumerate(words):
        if w == word:
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            if seed in words[start:end]:
                count += 1
    return count
