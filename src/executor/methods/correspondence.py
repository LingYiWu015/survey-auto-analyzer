"""对应分析 — CA / MCA 对应分析，感知图."""

import pandas as pd
import numpy as np
from typing import Tuple
import prince

from ...utils.data_utils import get_categorical_columns


def check_applicability(survey_def) -> Tuple[bool, str]:
    cat = get_categorical_columns(survey_def)
    if len(cat) < 2:
        return False, f"需要至少2个分类变量，当前 {len(cat)} 个"
    return True, ""


def run(survey_def, **kwargs) -> dict:
    """Run Correspondence Analysis (CA) or Multiple CA."""
    df = survey_def.df
    cat_cols = get_categorical_columns(survey_def)

    tables = []
    warnings = []

    # ── MCA for multiple categorical variables ──
    if len(cat_cols) >= 3:
        try:
            cat_data = df[cat_cols[:8]].dropna().astype(str)
            if len(cat_data) < 20:
                return {"tables": [], "charts": [], "interpretation": "有效样本不足", "warnings": ["样本<20"]}

            mca = prince.MCA(n_components=2, n_iter=10, random_state=42)
            mca = mca.fit(cat_data)

            # Eigenvalues
            eigenvalues = mca.eigenvalues_
            eig_rows = []
            for i, ev in enumerate(eigenvalues[:5]):
                eig_rows.append({
                    "维度": i + 1,
                    "特征值": round(ev, 4),
                    "解释方差%": f"{round(ev / sum(eigenvalues) * 100, 1)}%",
                })

            tables.append({"title": "MCA 特征值", "data": eig_rows})

            # Row coordinates
            row_coords = mca.row_coordinates(cat_data)
            coord_rows = []
            for idx, row in row_coords.head(30).iterrows():
                coord_rows.append({
                    "样本": str(idx)[:20],
                    "Dim1": round(row[0], 3),
                    "Dim2": round(row[1], 3),
                })
            tables.append({"title": "样本坐标（前30）", "data": coord_rows})

        except Exception as e:
            warnings.append(f"MCA分析异常: {e}")

    # ── Simple CA for two variables ──
    elif len(cat_cols) >= 2:
        try:
            col1, col2 = cat_cols[0], cat_cols[1]
            crosstab = pd.crosstab(df[col1], df[col2])

            ca = prince.CA(n_components=2, n_iter=10, random_state=42)
            ca = ca.fit(crosstab)

            # Column coordinates
            col_coords = ca.column_coordinates(crosstab)
            coord_rows = []
            for idx, row in col_coords.iterrows():
                coord_rows.append({
                    "类别": str(idx)[:30],
                    "Dim1": round(row[0], 3),
                    "Dim2": round(row[1], 3),
                })
            tables.append({"title": "CA 类别坐标", "data": coord_rows})

            # Eigenvalues
            eig_rows = []
            for i, ev in enumerate(ca.eigenvalues_[:5]):
                eig_rows.append({
                    "维度": i + 1,
                    "特征值": round(ev, 4),
                    "惯量%": f"{round(ev / sum(ca.eigenvalues_) * 100, 1)}%",
                })
            tables.append({"title": "CA 特征值", "data": eig_rows})

        except Exception as e:
            warnings.append(f"CA分析异常: {e}")

    interpretation = (
        "对应分析将分类变量的关联关系展现在二维感知图上。  \n"
        "距离越近的类别表示关联越强，距离越远表示关联越弱。  \n"
        "两个维度的惯量占比越高，表示二维图对原始关系的还原越好。"
    )

    return {
        "tables": tables,
        "charts": [],
        "interpretation": interpretation,
        "warnings": warnings,
    }
