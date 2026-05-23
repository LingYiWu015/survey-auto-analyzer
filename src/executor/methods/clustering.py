"""聚类分析 — K-Means 聚类，肘部法，轮廓系数."""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from ...utils.data_utils import get_numeric_columns, ensure_numeric


def check_applicability(survey_def) -> Tuple[bool, str]:
    num = get_numeric_columns(survey_def)
    if len(num) < 4:
        return False, f"需要至少4个连续/量表变量，当前 {len(num)} 个"
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Run K-Means clustering with elbow method and silhouette analysis."""
    df = survey_def.df
    num_cols = get_numeric_columns(survey_def)[:20]
    data = ensure_numeric(df, num_cols)[num_cols].dropna()

    if len(data) < 30:
        return {"tables": [], "charts": [], "interpretation": "样本量不足（<30）", "warnings": ["样本量<30"]}

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    tables = []
    warnings = []

    # ── Elbow method (K=2 to 10) ──
    max_k = min(10, len(data) // 5)
    inertias = []
    silhouettes = []

    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            inertias.append({"K": k, "惯性(Inertia)": round(km.inertia_, 1)})
            sil = silhouette_score(X, labels)
            silhouettes.append({"K": k, "轮廓系数": round(sil, 3)})
        except Exception:
            pass

    tables.append({"title": "肘部法 (Elbow Method)", "data": inertias})
    tables.append({"title": "轮廓系数 (Silhouette Score)", "data": silhouettes})

    # ── Best K (highest silhouette) ──
    if silhouettes:
        best_k = max(silhouettes, key=lambda x: x["轮廓系数"])["K"]
    else:
        best_k = 3

    # ── Run clustering with best K ──
    try:
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = []
        for u, c in zip(unique, counts):
            cluster_sizes.append({
                "聚类": f"类型{int(u)+1}",
                "样本数": int(c),
                "占比": f"{round(c / len(labels) * 100, 1)}%",
            })
        tables.append({"title": f"聚类结果 (K={best_k})", "data": cluster_sizes})

        # Cluster centers (on original scale)
        centers = scaler.inverse_transform(km.cluster_centers_)
        center_rows = []
        for i in range(best_k):
            row = {"聚类": f"类型{i+1}"}
            for j, col in enumerate(num_cols[:10]):
                row[col[:30]] = round(centers[i][j], 2)
            center_rows.append(row)
        tables.append({"title": "聚类中心（原始尺度）", "data": center_rows})

    except Exception as e:
        warnings.append(f"聚类分析异常: {e}")

    interpretation = (
        f"通过K-Means聚类将样本分为 {best_k} 类。  \n"
        f"最佳K值由轮廓系数确定（越接近1聚类效果越好）。  \n"
        f"聚类中心值反映了每类用户在各项指标上的平均水平，可用于用户画像刻画。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }
