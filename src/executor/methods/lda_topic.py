"""LDA 主题模型 — gensim LDA 主题提取."""

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


def run(survey_def, n_topics: int = 5, **kwargs) -> dict:
    """Run LDA topic modeling on text data."""
    df = survey_def.df
    text_cols = get_text_columns(survey_def)
    tables = []
    warnings = []

    try:
        import jieba
        from gensim import corpora, models
    except ImportError as e:
        return {
            "tables": [], "charts": [],
            "interpretation": f"依赖未安装: {e}",
            "warnings": [str(e)],
        }

    for col in text_cols:
        if col not in df.columns:
            continue

        texts = df[col].dropna().astype(str)
        texts = texts[texts.str.len() >= 10]  # Skip very short

        if len(texts) < 20:
            warnings.append(f"'{col[:40]}' 有效文本不足20条，跳过LDA")
            continue

        # Tokenize
        stopwords = _get_stopwords()
        tokenized = []
        for text in texts:
            words = jieba.lcut(text)
            words = [w.strip() for w in words if len(w.strip()) >= 2 and w.strip() not in stopwords]
            if words:
                tokenized.append(words)

        if len(tokenized) < 10:
            continue

        # Build dictionary and corpus
        dictionary = corpora.Dictionary(tokenized)
        dictionary.filter_extremes(no_below=2, no_above=0.8)
        corpus = [dictionary.doc2bow(text) for text in tokenized]

        # Train LDA
        actual_topics = min(n_topics, len(dictionary) // 10, 10)
        actual_topics = max(2, actual_topics)

        try:
            lda_model = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=actual_topics,
                random_state=42,
                passes=10,
                alpha="auto",
            )

            # Extract topic words
            topic_rows = []
            for topic_id in range(actual_topics):
                words = lda_model.show_topic(topic_id, topn=10)
                topic_words = ", ".join([w for w, _ in words])
                topic_rows.append({
                    "主题": f"主题{topic_id + 1}",
                    "关键词": topic_words,
                })

            tables.append({
                "title": f"LDA主题模型: {col[:50]} ({actual_topics}个主题)",
                "data": topic_rows,
            })

        except Exception as e:
            warnings.append(f"LDA建模异常 ({col[:30]}): {e}")

    if not tables:
        return {"tables": [], "charts": [], "interpretation": "文本数据不足以进行主题建模", "warnings": warnings}

    interpretation = (
        f"LDA（Latent Dirichlet Allocation）无监督地从文本中提取了潜在主题。  \n"
        f"每个主题由一组关键词表征，关键词权重越高表示越能代表该主题。  \n"
        f"建议根据关键词的语义为每个主题命名，以理解用户反馈的主要内容类别。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }


def _get_stopwords() -> set:
    """Get a basic Chinese stopword set."""
    stopwords = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
        "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
        "什么", "怎么", "如何", "为什么", "因为", "所以", "但是", "然而",
        "可以", "这个", "那个", "还是", "只是", "已经", "之后", "然后",
        "吧", "吗", "呢", "啊", "哦", "嗯", "哈",
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "it", "its", "and", "or", "but", "not", "no", "this", "that",
    }
    return stopwords
