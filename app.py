"""Survey Auto Analyzer — 智能问卷分析系统

Streamlit Web Application.
Upload survey data → Auto-detect question types → AI-powered method recommendations
→ Execute analyses → Download HTML report.

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional

# Import project modules
from src.config import config
from src.parser.survey_parser import parse_survey, get_analysis_summary
from src.engine.rule_engine import get_analysis_plan, summarize_plan
from src.engine.rules import SurveyContext
from src.ranker.scorer import score_methods
from src.ranker.dependency import resolve_stages
from src.executor.runner import AnalysisRunner, ExecutionReport
from src.reporter.generator import generate_draft_report, generate_full_report

# ─── Method Registry ─────────────────────────────────────────────────

def _build_method_registry() -> Dict:
    """Build the method registry for the execution runner."""
    from src.executor.methods import (
        reliability, descriptive, ttest_anova, correlation,
        regression, factor_analysis, clustering, correspondence,
        chi_square, runs_test, nps, sentiment, lda_topic,
        tfidf, so_pmi, sdgs, fuzzy_eval, ccsi_acsi, fsqca,
    )

    return {
        "reliability": reliability,
        "descriptive": descriptive,
        "ttest_anova": ttest_anova,
        "correlation": correlation,
        "regression": regression,
        "factor_analysis": factor_analysis,
        "clustering": clustering,
        "correspondence": correspondence,
        "chi_square": chi_square,
        "runs_test": runs_test,
        "nps": nps,
        "sentiment": sentiment,
        "lda_topic": lda_topic,
        "tfidf": tfidf,
        "so_pmi": so_pmi,
        "sdgs": sdgs,
        "fuzzy_eval": fuzzy_eval,
        "ccsi_acsi": ccsi_acsi,
        "fsqca": fsqca,
    }


# ─── Page Config ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Survey Auto Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a73e8; margin-bottom: 0; }
    .sub-header { color: #666; font-size: 0.9rem; margin-top: 0; }
    .method-card { background: #f8f9fa; border-radius: 8px; padding: 12px 16px; margin: 6px 0; border-left: 4px solid #1a73e8; }
    .method-card.foundational { border-left-color: #34a853; }
    .method-card.high { border-left-color: #1a73e8; }
    .method-card.medium { border-left-color: #fbbc04; }
    .method-card.low { border-left-color: #ea4335; }
    .score-badge { display: inline-block; padding: 4px 10px; border-radius: 12px; color: white; font-weight: 600; font-size: 0.85rem; }
    .success-box { background: #e8f5e9; border-radius: 8px; padding: 16px; }
    .error-box { background: #ffebee; border-radius: 8px; padding: 16px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────

_defaults = {
    "survey_def": None,
    "recommendations": None,
    "scored_methods": None,
    "llm_result": None,
    "execution_report": None,
    "selected_methods": [],
    "report_html": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Sidebar ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ 配置")

    # DeepSeek API Key
    api_key = st.text_input(
        "DeepSeek API Key",
        value=config.deepseek_api_key or "",
        type="password",
        help="从 https://platform.deepseek.com 获取",
    )
    if api_key:
        config.deepseek_api_key = api_key

    st.divider()

    # Output depth
    output_depth = st.radio(
        "📝 输出深度",
        options=["草稿（推荐+假设）", "成品（完整分析报告）"],
        index=0,
        help="草稿：仅方法推荐和AI假设 | 成品：执行所有分析并生成完整报告",
    )
    is_full_mode = "成品" in output_depth

    st.divider()

    # LLM toggle
    use_llm = st.checkbox(
        "🤖 启用 DeepSeek AI 语义分析",
        value=True,
        help="使用AI理解题目含义并生成研究假设",
    )

    st.divider()
    st.markdown("*Survey Auto Analyzer v1.0*")
    st.markdown("[📖 方法手册](../surveys_project/)")

    # ── History ──
    if "history" not in st.session_state:
        st.session_state.history = []
    if st.session_state.history:
        st.divider()
        st.markdown("#### 📜 历史记录")
        for h in st.session_state.history[-5:]:
            st.caption(f"📄 {h['name']} ({h['date']})")
            st.caption(f"   {h['methods']} 个方法 | {h['status']}")

# ─── Main Content ────────────────────────────────────────────────────

st.markdown('<p class="main-header">📊 Survey Auto Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">上传问卷数据 → 智能匹配分析方法 → 自动生成报告</p>', unsafe_allow_html=True)

# Step indicator
step = 1
if st.session_state.survey_def is not None:
    step = 2
if st.session_state.scored_methods is not None:
    step = 3
if st.session_state.execution_report is not None:
    step = 4

steps_text = ["📥 上传数据", "🧠 智能分析", "⚙️ 执行分析", "📄 生成报告"]
step_indicators = "  →  ".join(
    f"**{s}**" if i + 1 == step else s
    for i, s in enumerate(steps_text)
)
st.caption(step_indicators)
st.divider()

# ─── Step 1: Upload ──────────────────────────────────────────────────

st.markdown("### 📥 第一步：上传问卷数据")

uploaded_file = st.file_uploader(
    "拖拽或选择问卷数据文件",
    type=["csv", "xlsx", "xls"],
    help="支持 CSV、Excel 格式。兼容问卷星、腾讯问卷、Google Forms 导出格式。",
)

if uploaded_file:
    # Save to temp file
    import tempfile
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Track file change: clear analysis state when a new file is uploaded
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("_last_file_id") != file_id:
        # New file — reset analysis state
        for key in ["survey_def", "recommendations", "scored_methods",
                     "llm_result", "execution_report", "selected_methods",
                     "report_html", "_type_corrections"]:
            st.session_state[key] = None if key != "selected_methods" else []
        st.session_state["_last_file_id"] = file_id

    try:
        with st.spinner("🔍 正在解析问卷结构..."):
            survey_def = parse_survey(tmp_path)
            st.session_state.survey_def = survey_def

        # ─── Show parsing results ──────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📋 样本量", survey_def.sample_size)
        with col2:
            st.metric("❓ 题目数", survey_def.n_questions)
        with col3:
            st.metric("📂 来源平台", survey_def.platform.upper())

        st.markdown("#### 题目类型识别结果")
        summary_df = get_analysis_summary(survey_def)
        # Filter out metadata
        display_df = summary_df[summary_df["变量类型"] != "metadata"]
        st.dataframe(
            display_df[["题目", "题型", "唯一值数", "缺失率", "示例值"]],
            width='stretch',
            height=min(400, 35 * len(display_df) + 38),
        )

        # Editable type corrections
        corrections = st.session_state.get("_type_corrections", {})
        with st.expander("✏️ 修正题目类型（可选）"):
            for _, row in display_df.iterrows():
                col_name = row["题目"]
                current_type = row["变量类型"]
                # Use user's previous correction if available, else auto-detected
                saved_type = corrections.get(col_name, current_type)
                type_options = ["single_choice", "multi_choice", "likert_scale", "continuous", "text", "nps", "demographic"]
                type_labels = ["单选题", "多选题", "Likert量表", "连续数值", "开放文本", "NPS评分", "人口统计"]
                default_idx = type_options.index(saved_type) if saved_type in type_options else 0
                new_type = st.selectbox(
                    f"{col_name[:80]}",
                    options=type_options,
                    format_func=lambda x, labels=type_labels, opts=type_options: labels[opts.index(x)] if x in opts else x,
                    index=default_idx,
                    key=f"type_{col_name}",
                )
                corrections[col_name] = new_type
            st.session_state["_type_corrections"] = corrections

            if st.button("🔄 应用题型修正并重新分析", width='stretch'):
                # Apply corrections to survey_def
                for col_name, new_type in corrections.items():
                    q = survey_def.get_question(col_name)
                    if q:
                        old_type = q.var_type
                        q.var_type = new_type
                        # Update var_types_summary
                        if old_type in survey_def.var_types_summary:
                            survey_def.var_types_summary[old_type] -= 1
                        survey_def.var_types_summary[new_type] = survey_def.var_types_summary.get(new_type, 0) + 1
                # Clear analysis cache to force re-analysis with new types
                st.session_state.recommendations = None
                st.session_state.scored_methods = None
                st.session_state.llm_result = None
                st.session_state.execution_report = None
                st.session_state.report_html = None
                st.success("✅ 题型已修正！请重新点击「开始智能分析」")
                st.rerun()

    except Exception as e:
        st.error(f"❌ 文件解析失败: {e}")

# ─── Step 2: Analyze ─────────────────────────────────────────────────

if st.session_state.survey_def is not None:
    st.divider()
    st.markdown("### 🧠 第二步：智能分析")

    # ── Button: trigger analysis computation (only runs once per click) ──
    if st.button("🚀 开始智能分析", type="primary", width='stretch'):
        survey_def = st.session_state.survey_def
        llm_suggestions = None

        with st.spinner("🔧 规则引擎匹配分析方法..."):
            # LLM analysis (if enabled)
            if use_llm and config.deepseek_api_key:
                with st.spinner("🤖 DeepSeek AI 正在理解题目语义..."):
                    try:
                        from src.llm.deepseek_client import get_client

                        questions_for_llm = []
                        for q in survey_def.questions:
                            questions_for_llm.append({
                                "col_name": q.col_name,
                                "var_type": q.var_type,
                                "label": q.label,
                                "options": q.options[:10],
                            })

                        client = get_client()
                        llm_result = client.analyze_survey_semantics(questions_for_llm)
                        st.session_state.llm_result = llm_result

                        if llm_result:
                            llm_suggestions = llm_result.get("method_suggestions", [])
                            st.success(f"✅ AI 分析了 {len(llm_result.get('questions', []))} 道题目，生成了 {len(llm_result.get('hypotheses', []))} 个研究假设")
                    except Exception as e:
                        st.warning(f"⚠️ DeepSeek API 调用失败: {e}")

            # Get recommendations and score
            recommendations = get_analysis_plan(survey_def, llm_suggestions)
            st.session_state.recommendations = recommendations

            ctx = SurveyContext(survey_def)
            scored = score_methods(recommendations, ctx, llm_suggestions)
            st.session_state.scored_methods = scored

            # Initialize selected methods (only on first analysis)
            st.session_state.selected_methods = [
                rec.method_id for rec in scored if rec.total_score >= 50
            ]

    # ── Display: show results if analysis has been done (persists across reruns) ──
    if st.session_state.scored_methods is not None:
        survey_def = st.session_state.survey_def
        scored = st.session_state.scored_methods

        # ── Method recommendations as interactive table ──
        st.markdown("#### 🎯 推荐分析方法（按优先级排序）")

        # Build DataFrame for display
        method_rows = []
        for rec in scored:
            prereq_names = []
            for p in rec.prerequisites:
                info = None
                from src.engine.rules import get_method_info
                info = get_method_info(p)
                prereq_names.append(info.name if info else p)

            method_rows.append({
                "优先级": f"⭐{rec.total_score:.0f}" if rec.total_score >= 80 else (
                    f"📊{rec.total_score:.0f}" if rec.total_score >= 50 else f"🔍{rec.total_score:.0f}"
                ),
                "方法": rec.name,
                "类别": rec.category,
                "说明": rec.description,
                "数据适配": f"{rec.score_data_fit:.0f}/40",
                "语义相关": f"{rec.score_semantic:.0f}/30",
                "依赖": ", ".join(prereq_names) if prereq_names else "—",
            })

        method_df = pd.DataFrame(method_rows)
        st.dataframe(
            method_df,
            width='stretch',
            height=min(400, 35 * len(method_df) + 38),
            hide_index=True,
            column_config={
                "优先级": st.column_config.TextColumn("优先级", width="small"),
                "方法": st.column_config.TextColumn("方法", width="medium"),
                "类别": st.column_config.TextColumn("类别", width="small"),
                "说明": st.column_config.TextColumn("说明", width="large"),
                "数据适配": st.column_config.TextColumn("数据适配", width="small"),
                "语义相关": st.column_config.TextColumn("语义相关", width="small"),
                "依赖": st.column_config.TextColumn("前置依赖", width="medium"),
            },
        )

        # ── Dependency visualization ──
        from src.ranker.dependency import resolve_stages, validate_plan
        all_method_ids = [s.method_id for s in scored]
        stages = resolve_stages(all_method_ids)
        plan_warnings = validate_plan(all_method_ids)

        with st.expander("📊 方法依赖关系与执行顺序"):
            if plan_warnings:
                for w in plan_warnings:
                    st.warning(w)
            for stage in stages:
                method_names = []
                for mid in stage.methods:
                    info = None
                    from src.engine.rules import get_method_info
                    info = get_method_info(mid)
                    method_names.append(info.name if info else mid)
                arrow = "→" if stage.stage_id > 1 else ""
                st.markdown(f"**阶段 {stage.stage_id}:** {arrow} {' | '.join(method_names)}")
            st.caption("同阶段方法可并行执行；箭头表示依赖顺序")

        # ── Method selection (persistent via session_state) ──
        st.markdown("#### ✅ 选择要执行的分析方法")
        method_options = {rec.method_id: f"{rec.name} ({rec.total_score:.0f}分)" for rec in scored}

        # Use session_state to persist selections across reruns
        selected = st.multiselect(
            "勾选要运行的方法（可多选）",
            options=list(method_options.keys()),
            default=st.session_state.get("selected_methods", []),
            format_func=lambda x: method_options.get(x, x),
            key="_method_multiselect",
        )
        st.session_state.selected_methods = selected

        # ── LLM hypotheses ──
        if st.session_state.llm_result:
            with st.expander("💡 查看 AI 生成的研究假设"):
                hypotheses = st.session_state.llm_result.get("hypotheses", [])
                if hypotheses:
                    for i, h in enumerate(hypotheses, 1):
                        conf = h.get("confidence", 0.5)
                        st.markdown(f"""
                        **假设{i}:** {h.get('text', '')}
                        *推荐方法: {h.get('recommended_method', '')} | 置信度: {conf:.0%}*
                        """)
                else:
                    st.info("AI 未生成研究假设")

        # ── Step 3: Execute (full mode) ──
        if is_full_mode and selected:
            st.divider()
            st.markdown("### ⚙️ 第三步：执行分析")

            if st.button("▶️ 开始执行分析", type="primary", width='stretch'):
                registry = _build_method_registry()
                runner = AnalysisRunner(registry)

                # Progress tracking with per-method status
                progress_bar = st.progress(0)
                status_text = st.empty()
                method_status_container = st.empty()

                def progress_callback(status, method_id, current, total):
                    progress_bar.progress(current / max(total, 1))
                    from src.engine.rules import get_method_info
                    info = get_method_info(method_id)
                    name = info.name if info else method_id
                    status_text.text(f"⏳ 执行中: {name} ({current}/{total})")

                with st.spinner("正在执行分析..."):
                    report = runner.run(
                        survey_def,
                        method_ids=selected,
                        progress_callback=progress_callback,
                    )
                    st.session_state.execution_report = report

                progress_bar.progress(1.0)
                status_text.text("✅ 分析完成！")

            # ── Show execution results (persists across reruns) ──
            if st.session_state.execution_report is not None:
                report = st.session_state.execution_report

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ 成功", report.n_success)
                with col2:
                    st.metric("❌ 失败", report.n_failed)
                with col3:
                    st.metric("⏱️ 总耗时", f"{report.total_duration:.1f}s")

                with st.expander("📋 查看执行详情"):
                    for result in report.results:
                        icon = {"success": "✅", "failed": "❌", "skipped": "⏭️", "not_applicable": "⊘"}.get(result.status, "❓")
                        if result.status == "success":
                            st.success(f"{icon} {result.method_name} ({result.duration_seconds:.1f}s)")
                        else:
                            st.error(f"{icon} {result.method_name}: {result.error_message}")

                # ── Report generation ──
                st.divider()
                st.markdown("### 📄 第四步：生成报告")

                if st.button("📥 生成完整报告", type="primary", width='stretch'):
                    with st.spinner("正在生成报告..."):
                        html = generate_full_report(
                            survey_def,
                            report,
                            st.session_state.recommendations,
                            {},
                        )
                        st.session_state.report_html = html
                    st.success("报告生成完成！")

                    # Record to history
                    st.session_state.history.append({
                        "name": Path(uploaded_file.name).stem if uploaded_file else "Unknown",
                        "date": datetime.now().strftime("%m-%d %H:%M"),
                        "methods": f"{report.n_success}/{len(report.results)}",
                        "status": "完成" if report.n_failed == 0 else f"{report.n_failed}项失败",
                    })

                if st.session_state.report_html:
                    st.download_button(
                        label="⬇️ 下载 HTML 报告",
                        data=st.session_state.report_html,
                        file_name=f"survey_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        width='stretch',
                    )
                    with st.expander("👁️ 报告预览"):
                        import urllib.parse
                        encoded = urllib.parse.quote(st.session_state.report_html.encode("utf-8"))
                        data_uri = f"data:text/html;charset=utf-8,{encoded}"
                        st.iframe(src=data_uri, height=600, width='stretch')
        if not is_full_mode:
            st.divider()
            st.markdown("### 📄 生成草稿报告")

            if st.button("📝 生成草稿报告", type="primary", width='stretch'):
                hypotheses = None
                if st.session_state.llm_result:
                    hypotheses = st.session_state.llm_result.get("hypotheses", [])

                with st.spinner("正在生成草稿报告..."):
                    html = generate_draft_report(
                        survey_def,
                        st.session_state.recommendations,
                        hypotheses,
                    )
                    st.session_state.report_html = html
                st.success("草稿报告生成完成！")

            if st.session_state.report_html:
                st.download_button(
                    label="⬇️ 下载草稿报告",
                    data=st.session_state.report_html,
                    file_name=f"survey_draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    width='stretch',
                )
                with st.expander("👁️ 预览草稿报告"):
                    import urllib.parse
                    encoded = urllib.parse.quote(st.session_state.report_html.encode("utf-8"))
                    data_uri = f"data:text/html;charset=utf-8,{encoded}"
                    st.iframe(src=data_uri, height=600, width='stretch')

# ─── Footer ──────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<p style='text-align:center; color:#999; font-size:0.8rem;'>"
    "Survey Auto Analyzer v1.0 | Powered by Python + Streamlit + DeepSeek AI | "
    "仅本地运行，数据不会上传到任何服务器"
    "</p>",
    unsafe_allow_html=True,
)
