"""Report Generator — generates HTML analysis reports.

Two modes:
- Draft: Method recommendations + AI hypotheses only
- Full: Complete analysis results with charts and interpretations
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)


def generate_draft_report(
    survey_def,
    recommendations: List,
    hypotheses: Optional[List[Dict]] = None,
    method_suggestions: Optional[List[Dict]] = None,
) -> str:
    """Generate a draft report (lightweight: recommendations + hypotheses).

    Returns HTML string.
    """
    # Build recommendations table
    rec_rows = ""
    for i, rec in enumerate(recommendations[:15], 1):
        star = "⭐" if rec.final_priority >= 80 else ("📊" if rec.final_priority >= 50 else "🔍")
        reasons = "<br>".join(rec.trigger_reasons[:3])
        rec_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{star} {rec.name}</td>
            <td>{rec.category}</td>
            <td><span class="score">{rec.final_priority}</span></td>
            <td>{rec.description}</td>
            <td>{reasons}</td>
        </tr>"""

    # Build hypotheses list
    hypo_html = ""
    if hypotheses:
        for h in hypotheses[:10]:
            confidence = h.get("confidence", 0.5)
            conf_color = "green" if confidence >= 0.8 else ("orange" if confidence >= 0.5 else "red")
            hypo_html += f"""
            <li>
                <strong>{h.get('text', '')}</strong>
                <span class="badge" style="background:{conf_color}">置信度: {confidence:.0%}</span>
                <br><small>推荐方法: {h.get('recommended_method', '')}</small>
            </li>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>问卷分析报告（草稿）</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; color: #333; }}
        h1 {{ color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; }}
        h2 {{ color: #444; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 14px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .score {{ font-weight: bold; color: #1a73e8; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; color: white; font-size: 12px; margin-left: 8px; }}
        .meta {{ color: #888; font-size: 13px; }}
        ul {{ line-height: 1.8; }}
    </style>
</head>
<body>
    <h1>📋 问卷分析报告（草稿）</h1>
    <p class="meta">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 样本量: {survey_def.sample_size} | 题目数: {survey_def.n_questions}</p>

    <h2>📊 样本概况</h2>
    <p>本次调查共回收 <strong>{survey_def.sample_size}</strong> 份有效问卷，包含 <strong>{survey_def.n_questions}</strong> 道题目。</p>
    <p>题型分布: {_format_var_types(survey_def.var_types_summary)}</p>

    <h2>🎯 推荐分析方法</h2>
    <p>根据题目类型和数据结构，推荐以下分析方法（按优先级排序）：</p>
    <table>
        <tr><th>#</th><th>方法</th><th>类别</th><th>优先级</th><th>说明</th><th>触发原因</th></tr>
        {rec_rows}
    </table>

    <h2>💡 AI 研究假设</h2>
    <p>基于题目语义分析，DeepSeek AI 生成以下研究假设：</p>
    <ol>{hypo_html if hypo_html else '<li>（未调用 AI 分析或无可生成假设）</li>'}</ol>

    <p class="meta" style="margin-top:40px;">报告由 Survey Auto Analyzer 自动生成 | 此为草稿版，仅含方法推荐与假设</p>
</body>
</html>"""

    return html


def generate_full_report(
    survey_def,
    execution_report,
    recommendations: List = None,
    llm_interpretations: Dict[str, str] = None,
) -> str:
    """Generate a full analysis report with all results.

    Returns HTML string.
    """
    results_html = ""

    for result in execution_report.results:
        status_icon = {"success": "✅", "failed": "❌", "skipped": "⏭️", "not_applicable": "⊘"}.get(result.status, "❓")

        if result.status == "success":
            tables_html = _format_tables(result.tables)
            charts_html = _format_charts(result.charts)

            # LLM interpretation if available
            llm_interp = ""
            if llm_interpretations and result.method_id in llm_interpretations:
                llm_interp = f'<div class="llm-interp">🤖 <strong>AI解读:</strong> {llm_interp}</div>'

            results_html += f"""
            <div class="method-result">
                <h3>{status_icon} {result.method_name} <span class="duration">({result.duration_seconds:.1f}s)</span></h3>
                {tables_html}
                {charts_html}
                {llm_interp}
                <div class="interpretation">{result.interpretation}</div>
            </div>"""
        else:
            results_html += f"""
            <div class="method-result failed">
                <h3>{status_icon} {result.method_name}</h3>
                <p class="error">{result.error_message}</p>
            </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>问卷分析报告（完整版）</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; color: #333; background: #fafafa; }}
        h1 {{ color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 15px; }}
        h2 {{ color: #333; margin-top: 35px; border-left: 4px solid #1a73e8; padding-left: 12px; }}
        h3 {{ color: #555; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0 20px 0; font-size: 13px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #e0e0e0; padding: 8px 12px; text-align: left; }}
        th {{ background: #f0f4ff; font-weight: 600; }}
        .method-result {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
        .method-result.failed {{ opacity: 0.7; background: #fff5f5; }}
        .duration {{ color: #999; font-size: 12px; font-weight: normal; }}
        .interpretation {{ background: #f8f9fa; padding: 12px 16px; border-radius: 6px; margin-top: 12px; line-height: 1.8; }}
        .llm-interp {{ background: #e8f5e9; padding: 10px 14px; border-radius: 6px; margin: 10px 0; }}
        .error {{ color: #d32f2f; }}
        .meta {{ color: #888; font-size: 13px; }}
        .summary-box {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
        .summary-box table {{ box-shadow: none; }}
    </style>
</head>
<body>
    <h1>📊 问卷分析报告（完整版）</h1>
    <p class="meta">
        生成时间: {execution_report.executed_at} |
        样本量: {survey_def.sample_size} |
        成功: {execution_report.n_success} |
        失败: {execution_report.n_failed} |
        总耗时: {execution_report.total_duration:.1f}s
    </p>

    <div class="summary-box">
        <h2>📋 执行摘要</h2>
        <p>本次分析共执行了 {len(execution_report.results)} 个分析方法，其中 {execution_report.n_success} 个成功完成。</p>
        <p>成功率: {execution_report.success_rate:.0f}%</p>
    </div>

    <h2>📈 分析结果</h2>
    {results_html}

    <p class="meta" style="margin-top:50px; text-align:center;">
        报告由 <strong>Survey Auto Analyzer</strong> 自动生成 |
        Powered by Python + Streamlit + DeepSeek AI
    </p>
</body>
</html>"""

    return html


def _format_tables(tables: List[Dict]) -> str:
    """Format analysis tables as HTML."""
    if not tables:
        return ""
    html = ""
    for t in tables:
        data = t.get("data", {})
        title = t.get("title", "")
        if isinstance(data, dict):
            # Key-value table
            html += f'<p><strong>{title}</strong></p><table>'
            for k, v in data.items():
                html += f'<tr><td>{k}</td><td>{v}</td></tr>'
            html += '</table>'
        elif isinstance(data, list) and len(data) > 0:
            # List of dicts table
            html += f'<p><strong>{title}</strong></p><table><tr>'
            for key in data[0].keys():
                html += f'<th>{key}</th>'
            html += '</tr>'
            for row in data[:30]:  # Limit rows
                html += '<tr>'
                for val in row.values():
                    html += f'<td>{val}</td>'
                html += '</tr>'
            html += '</table>'
    return html


def _format_charts(charts: List[Dict]) -> str:
    """Format chart data for HTML embedding with Plotly JS."""
    if not charts:
        return ""
    html_parts = []
    has_plotly = any(c.get("type") == "plotly" for c in charts)
    if has_plotly:
        html_parts.append(
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
        )

    for i, chart in enumerate(charts):
        chart_type = chart.get("type", "")
        if chart_type == "plotly" and chart.get("figure"):
            div_id = f"plotly_chart_{i}"
            figure_json = chart["figure"]
            title = chart.get("title", "")
            # Use a script tag to store JSON, then render with Plotly
            html_parts.append(
                f'<div id="{div_id}" style="width:100%;min-height:400px;"></div>'
                f'<script type="application/json" id="{div_id}_data">'
                f'{figure_json}'
                f'</script>'
                f'<script>'
                f'(function(){{'
                f'var el=document.getElementById("{div_id}_data");'
                f'if(el && window.Plotly){{'
                f'var fig=JSON.parse(el.textContent);'
                f'Plotly.newPlot("{div_id}", fig.data, fig.layout);'
                f'}}'
                f'}})();'
                f'</script>'
            )

    return "\n".join(html_parts)


def _format_var_types(var_types: Dict[str, int]) -> str:
    """Format variable type summary."""
    labels = {
        "single_choice": "单选题",
        "multi_choice": "多选题",
        "likert_scale": "Likert量表",
        "continuous": "连续数值",
        "text": "开放文本",
        "nps": "NPS评分",
        "demographic": "人口统计",
    }
    parts = []
    for k, v in var_types.items():
        label = labels.get(k, k)
        parts.append(f"{label}×{v}")
    return ", ".join(parts) if parts else "无"
