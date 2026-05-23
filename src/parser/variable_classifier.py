"""Variable Classifier - automatically classifies survey question types.

Detects variable types from column data:
- single_choice: categorical with few unique values
- multi_choice: binary (0/1) columns belonging to a multi-select question group
- likert_scale: ordinal scale with typical Likert patterns
- continuous: numeric data
- text: free-text responses
- nps: Net Promoter Score (0-10 scale)
- demographic: age/gender/income etc.
"""

import re
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Likert detection keywords in column names
_LIKERT_KEYWORDS = [
    "满意", "同意", "重要", "需要", "兴趣", "认可", "信任",
    "质量", "服务", "体验", "期望", "感知", "态度", "意愿",
    "评价", "评分", "打分", "程度", "水平", "状况",
    "satisf", "agree", "important", "quality", "rate", "score",
]

# Demographic detection keywords
_DEMO_KEYWORDS = [
    "性别", "年龄", "学历", "教育", "收入", "职业", "行业",
    "地区", "城市", "省份", "婚姻", "家庭", "子女",
    "gender", "age", "education", "income", "occupation",
]

# NPS detection keywords
_NPS_KEYWORDS = [
    "推荐", "nps", "净推荐值", "0-10", "0-10分",
    "promoter", "recommend",
]

# Likert value patterns
_LIKERT_VALUE_PATTERNS = [
    # Chinese standard 5-point
    {"非常不满意", "不满意", "一般", "满意", "非常满意"},
    {"非常不同意", "不同意", "中立", "同意", "非常同意"},
    {"非常不赞同", "不赞同", "一般", "赞同", "非常赞同"},
    {"从不", "很少", "有时", "经常", "总是"},
    {"非常低", "较低", "中等", "较高", "非常高"},
    {"非常不重要", "不重要", "一般", "重要", "非常重要"},
    # English
    {"strongly disagree", "disagree", "neutral", "agree", "strongly agree"},
    {"very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"},
    {"never", "rarely", "sometimes", "often", "always"},
]


def _normalize_text(s: str) -> str:
    """Normalize text for comparison."""
    return s.strip().lower().replace(" ", "")


def _is_likert_values(unique_values: pd.Series) -> bool:
    """Check if unique values match any known Likert pattern.

    Returns True if at least 60% of unique values match a Likert pattern.
    """
    norm_vals = set(_normalize_text(str(v)) for v in unique_values.dropna())

    for pattern in _LIKERT_VALUE_PATTERNS:
        norm_pattern = {_normalize_text(p) for p in pattern}
        overlap = norm_vals & norm_pattern
        if len(overlap) >= min(len(norm_pattern), len(norm_vals)) * 0.6:
            return True

    # Also check if values are numeric 1-5, 1-7, 0-10
    numeric_vals = pd.to_numeric(unique_values, errors="coerce")
    if numeric_vals.notna().all():
        valid_nums = numeric_vals.dropna()
        if len(valid_nums) > 0:
            mn, mx = valid_nums.min(), valid_nums.max()
            if mn >= 0 and mx <= 10 and len(valid_nums.unique()) <= 11:
                return True

    return False


def _column_name_has(pattern_list: List[str], col_name: str) -> bool:
    """Check if column name contains any of the pattern keywords."""
    name_lower = col_name.lower()
    return any(kw in name_lower for kw in pattern_list)


def classify_column(
    col_name: str, series: pd.Series, all_columns: Optional[List[str]] = None
) -> str:
    """Classify a single column into a variable type.

    Args:
        col_name: The column header / question text.
        series: The data for this column.
        all_columns: List of all column names (for multi-choice detection).

    Returns one of:
        'single_choice', 'multi_choice', 'likert_scale',
        'continuous', 'text', 'nps', 'demographic', 'metadata', 'unknown'
    """
    # 1. Detect metadata columns (submission time, IP, serial number, etc.)
    meta_patterns = [
        r"^(序号|编号|id|no\.?\d*)$",
        r"(提交时间|开始时间|结束时间|答卷时长|所用时间|timestamp|time)",
        r"(ip|ip地址|ipaddr|来源|渠道|browser|浏览器)",
        r"(审核|审核状态|是否有效|标记)",
    ]
    for pat in meta_patterns:
        if re.search(pat, col_name.lower()):
            return "metadata"

    # Drop fully null
    non_null = series.dropna()
    if len(non_null) == 0:
        return "metadata"

    unique_vals = non_null.unique()
    n_unique = len(unique_vals)
    n_total = len(non_null)
    unique_ratio = n_unique / max(n_total, 1)

    # 2. Detect numeric / continuous
    numeric_series = pd.to_numeric(non_null, errors="coerce")
    is_fully_numeric = numeric_series.notna().all()

    if is_fully_numeric:
        # Check if NPS (0-10 scale with NPS keywords)
        mn, mx = numeric_series.min(), numeric_series.max()
        if _column_name_has(_NPS_KEYWORDS, col_name) and mn >= 0 and mx <= 10:
            return "nps"

        # Likert if range is small and keywords match
        if mx - mn <= 10 and n_unique <= 11:
            if _column_name_has(_LIKERT_KEYWORDS, col_name) or mx - mn <= 5:
                return "likert_scale"

        # Check for binary (0/1) - could be multi-choice
        if set(unique_vals).issubset({0, 1, "0", "1"}) and n_unique <= 2:
            if all_columns and _has_sibling_binary(col_name, all_columns):
                return "multi_choice"
            return "single_choice"

        return "continuous"

    # 3. Detect text / open-ended
    avg_len = non_null.astype(str).str.len().mean()
    if avg_len > 30 or n_unique / max(n_total, 1) > 0.8:
        return "text"

    # 4. Detect categorical (single_choice / multi_choice)
    if n_unique <= 20:
        # Check if it looks like Likert values
        if _is_likert_values(non_null):
            return "likert_scale"

        # Check if binary with siblings → multi_choice component
        if n_unique <= 2:
            if all_columns and _has_sibling_binary(col_name, all_columns):
                return "multi_choice"

        # Demographic check
        if _column_name_has(_DEMO_KEYWORDS, col_name):
            return "demographic"

        return "single_choice"

    # 5. Fallback
    return "text" if avg_len > 20 else "single_choice"


def _has_sibling_binary(col_name: str, all_columns: List[str]) -> bool:
    """Check if there are other columns that look like part of the same
    multi-choice question (similar prefix, binary values).

    Multi-choice questions are often exported as:
      1.您使用的品牌_微信, 1.您使用的品牌_支付宝, 1.您使用的品牌_银联
    """
    # Extract base name (remove trailing _option or -option)
    base = re.sub(r"[_\-—].{1,20}$", "", col_name.strip())
    n_siblings = 0
    for c in all_columns:
        if c != col_name:
            c_base = re.sub(r"[_\-—].{1,20}$", "", c.strip())
            if c_base == base:
                n_siblings += 1
    return n_siblings >= 1


def classify_all_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Classify all columns in the DataFrame.

    Returns dict: {column_name: variable_type}
    """
    all_cols = list(df.columns)
    result = {}

    for col in all_cols:
        result[col] = classify_column(col, df[col], all_cols)

    # Second pass: group multi_choice columns together
    multi_cols = [c for c, t in result.items() if t == "multi_choice"]
    _group_multi_choice(multi_cols, result)

    return result


def _group_multi_choice(multi_cols: List[str], result: Dict[str, str]):
    """Group multi_choice columns by their base question name.

    Detects patterns like:
      "5.渠道_微信", "5.渠道_支付宝", "5.渠道_银联"
    or
      "Q3（多选）_选项A", "Q3（多选）_选项B"

    Marks columns with the same base question as a group by storing their
    group ID in a class-level registry (for later use by the executor).
    """
    if len(multi_cols) < 2:
        return

    # Extract base names and group
    groups: Dict[str, List[str]] = {}
    for col in multi_cols:
        base = re.sub(r"[_\-—].{1,20}$", "", col.strip())
        # Also try removing parenthesized suffixes like （多选）
        base = re.sub(r"[（(][^)）]*[)）]$", "", base).strip()
        if base not in groups:
            groups[base] = []
        groups[base].append(col)

    # Only keep groups with 2+ columns
    for base, cols in groups.items():
        if len(cols) >= 2:
            # Mark all columns in this group with a group tag
            for col in cols:
                # Keep as multi_choice but could add group info
                pass


def get_variable_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary DataFrame of classified variables.

    Returns DataFrame with columns:
        column_name, variable_type, n_unique, n_missing, missing_rate, sample_values
    """
    classification = classify_all_columns(df)
    rows = []

    for col in df.columns:
        var_type = classification.get(col, "unknown")
        series = df[col]
        n_missing = series.isna().sum()
        n_total = len(series)
        non_null = series.dropna()
        n_unique = non_null.nunique()
        sample_vals = ", ".join(
            str(v) for v in non_null.unique()[:5]
        )

        rows.append({
            "题目": col,
            "题型": _type_label(var_type),
            "变量类型": var_type,
            "唯一值数": n_unique,
            "缺失数": n_missing,
            "缺失率": f"{n_missing / max(n_total, 1) * 100:.1f}%",
            "示例值": sample_vals[:80],
        })

    return pd.DataFrame(rows)


def _type_label(var_type: str) -> str:
    """Get human-readable Chinese label for variable type."""
    labels = {
        "single_choice": "单选题",
        "multi_choice": "多选题",
        "likert_scale": "Likert量表",
        "continuous": "连续数值",
        "text": "开放文本",
        "nps": "NPS评分",
        "demographic": "人口统计",
        "metadata": "元数据",
        "unknown": "未知",
    }
    return labels.get(var_type, var_type)
