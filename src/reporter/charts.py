"""Chart utilities — generates Plotly charts for the analysis report.

Handles Chinese font configuration for matplotlib and creates
interactive Plotly charts for various analysis types.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Try to configure Chinese font support
try:
    import matplotlib
    import matplotlib.pyplot as plt
    # Try common Chinese fonts
    for font_name in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]:
        try:
            matplotlib.font_manager.findfont(font_name, fallback_to_default=False)
            matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue
except ImportError:
    pass


def create_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "相关系数热力图",
) -> dict:
    """Create a correlation heatmap chart data."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
        ))
        fig.update_layout(title=title, height=600)
        return {"type": "plotly", "figure": fig.to_json(), "title": title}
    except Exception as e:
        logger.warning(f"Heatmap creation failed: {e}")
        return {"type": "error", "message": str(e)}


def create_bar_chart(
    labels: List[str],
    values: List[float],
    title: str = "柱状图",
    xlabel: str = "",
    ylabel: str = "",
) -> dict:
    """Create a bar chart data."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Bar(x=labels, y=values))
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
        return {"type": "plotly", "figure": fig.to_json(), "title": title}
    except Exception as e:
        logger.warning(f"Bar chart creation failed: {e}")
        return {"type": "error", "message": str(e)}


def create_pie_chart(
    labels: List[str],
    values: List[float],
    title: str = "饼图",
) -> dict:
    """Create a pie chart data."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Pie(labels=labels, values=values))
        fig.update_layout(title=title)
        return {"type": "plotly", "figure": fig.to_json(), "title": title}
    except Exception as e:
        logger.warning(f"Pie chart creation failed: {e}")
        return {"type": "error", "message": str(e)}


def create_radar_chart(
    categories: List[str],
    values: List[float],
    title: str = "雷达图",
) -> dict:
    """Create a radar chart data."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
        ))
        fig.update_layout(title=title)
        return {"type": "plotly", "figure": fig.to_json(), "title": title}
    except Exception as e:
        logger.warning(f"Radar chart creation failed: {e}")
        return {"type": "error", "message": str(e)}


def create_elbow_plot(
    k_values: List[int],
    inertias: List[float],
    title: str = "肘部法 (Elbow Method)",
) -> dict:
    """Create an elbow plot for K-Means."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values, y=inertias, mode="lines+markers",
            marker=dict(size=10),
        ))
        fig.update_layout(
            title=title,
            xaxis_title="K (聚类数)",
            yaxis_title="惯性 (Inertia)",
        )
        return {"type": "plotly", "figure": fig.to_json(), "title": title}
    except Exception as e:
        logger.warning(f"Elbow plot creation failed: {e}")
        return {"type": "error", "message": str(e)}


def create_wordcloud_data(
    words: Dict[str, float],
    title: str = "词云",
) -> dict:
    """Generate word cloud data for front-end rendering.

    Returns word-frequency pairs for client-side word cloud rendering.
    """
    sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:100]
    return {
        "type": "wordcloud",
        "title": title,
        "data": [{"text": w, "value": s} for w, s in sorted_words],
    }


def matplotlib_to_plotly(fig) -> dict:
    """Convert a matplotlib figure to plotly JSON."""
    try:
        import plotly.tools as tls
        py_fig = tls.mpl_to_plotly(fig)
        return {"type": "plotly", "figure": py_fig.to_json()}
    except Exception:
        return {"type": "error", "message": "Failed to convert matplotlib figure"}
