"""TF-IDF 关键词提取."""

import pandas as pd
import numpy as np
from typing import Tuple

from ...utils.data_utils import get_text_columns


def check_applicability(survey_def) -> Tuple[bool, str]:
    text_cols = get_text_columns(survey_def)
    if not text_cols:
        return False, "未检测到开放文本题"
    return True, ""


def run(survey_def, top_n: int = 20, **kwargs) -> dict:
    """Extract keywords using TF-IDF."""
    df = survey_def.df
    text_cols = get_text_columns(survey_def)
    tables = []
    warnings = []

    try:
        import jieba
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as e:
        return {"tables": [], "charts": [], "interpretation": f"依赖未安装: {e}", "warnings": [str(e)]}

    for col in text_cols:
        if col not in df.columns:
            continue

        texts = df[col].dropna().astype(str)
        texts = texts[texts.str.len() >= 5]

        if len(texts) < 3:
            continue

        # Tokenize with jieba
        tokenized = []
        for text in texts:
            words = jieba.lcut(text)
            words = [w.strip() for w in words if len(w.strip()) >= 2]
            tokenized.append(" ".join(words))

        try:
            vectorizer = TfidfVectorizer(max_features=top_n, token_pattern=r"(?u)\b\w+\b")
            tfidf_matrix = vectorizer.fit_transform(tokenized)
            feature_names = vectorizer.get_feature_names_out()

            # Get mean TF-IDF per term
            mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            term_scores = sorted(
                zip(feature_names, mean_tfidf),
                key=lambda x: x[1], reverse=True
            )

            keyword_rows = []
            for rank, (term, score) in enumerate(term_scores[:top_n], 1):
                keyword_rows.append({
                    "排名": rank,
                    "关键词": term,
                    "TF-IDF均值": round(score, 5),
                })

            tables.append({
                "title": f"TF-IDF关键词: {col[:50]}",
                "data": keyword_rows,
            })

        except Exception as e:
            warnings.append(f"TF-IDF异常 ({col[:30]}): {e}")

    if not tables:
        return {"tables": [], "charts": [], "interpretation": "无有效文本数据", "warnings": warnings}

    interpretation = (
        "TF-IDF（词频-逆文档频率）衡量词语对文档的重要程度。  \n"
        "高频且在不同回复中出现较少的词得分更高，代表用户反馈中的关键信息。  \n"
        "建议结合词云进行可视化展示。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }
