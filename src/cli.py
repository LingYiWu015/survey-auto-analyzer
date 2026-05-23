"""CLI entry point for Survey Auto Analyzer.

Usage:
    survey-auto analyze <datafile> [options]
    survey-auto --list-methods
    survey-auto --version

Examples:
    survey-auto analyze survey_data.xlsx --draft
    survey-auto analyze survey_data.xlsx -o report.html --methods reliability,descriptive,nps
    survey-auto analyze survey_data.csv --api-key sk-xxx --no-streamlit
"""

import argparse
import sys
from pathlib import Path

# Ensure the package root is importable
_package_root = Path(__file__).parent.parent
sys.path.insert(0, str(_package_root))


def cmd_analyze(args):
    """Run survey analysis from the command line."""
    from src.parser.survey_parser import parse_survey
    from src.engine.rule_engine import get_analysis_plan
    from src.engine.rules import SurveyContext
    from src.ranker.scorer import score_methods
    from src.executor.runner import AnalysisRunner
    from src.executor.methods import (
        reliability, descriptive, ttest_anova, correlation,
        regression, factor_analysis, clustering, correspondence,
        chi_square, runs_test, nps, sentiment, lda_topic,
        tfidf, so_pmi, sdgs, fuzzy_eval, ccsi_acsi, fsqca,
    )
    from src.reporter.generator import generate_draft_report, generate_full_report

    filepath = args.datafile
    if not Path(filepath).exists():
        print(f"❌ 文件不存在: {filepath}")
        sys.exit(1)

    # 1. Parse
    print("🔍 解析问卷结构...")
    survey = parse_survey(filepath)
    print(f"   样本量: {survey.sample_size}, 题目数: {survey.n_questions}")
    print(f"   题型分布: {survey.var_types_summary}")

    # 2. Rule engine
    print("\n🧠 规则引擎匹配分析方法...")
    llm_suggestions = None

    if not args.no_llm:
        api_key = args.api_key or None
        if api_key:
            try:
                from src.config import config
                config.deepseek_api_key = api_key
                from src.llm.deepseek_client import get_client

                questions_for_llm = []
                for q in survey.questions:
                    questions_for_llm.append({
                        "col_name": q.col_name,
                        "var_type": q.var_type,
                        "label": q.label,
                        "options": q.options[:10],
                    })

                client = get_client()
                llm_result = client.analyze_survey_semantics(questions_for_llm)
                if llm_result:
                    llm_suggestions = llm_result.get("method_suggestions", [])
                    hypotheses = llm_result.get("hypotheses", [])
                    print(f"   AI 生成了 {len(hypotheses)} 个研究假设")
            except Exception as e:
                print(f"   ⚠️ LLM 调用失败: {e}")

    recommendations = get_analysis_plan(survey, llm_suggestions)
    ctx = SurveyContext(survey)
    scored = score_methods(recommendations, ctx, llm_suggestions)

    # Show method ranking
    print("\n📊 推荐分析方法:")
    for i, s in enumerate(scored[:10], 1):
        star = "⭐" if s.total_score >= 80 else ("📊" if s.total_score >= 50 else "🔍")
        print(f"   {i:2}. {star} {s.name:20s}  ({s.total_score:.0f}分) — {s.description}")

    # 3. Draft mode: just recommendations
    if args.draft:
        print("\n📝 生成草稿报告...")
        html = generate_draft_report(
            survey,
            recommendations,
            llm_result.get("hypotheses", []) if 'llm_result' in dir() else None,
        )
        output_path = args.output or filepath.replace(".xlsx", "_draft.html").replace(".csv", "_draft.html")
        Path(output_path).write_text(html, encoding="utf-8")
        print(f"   ✅ 报告已保存: {output_path}")
        return

    # 4. Full mode: execute analyses
    method_ids = args.methods.split(",") if args.methods else [
        s.method_id for s in scored if s.total_score >= 50
    ]

    print(f"\n⚙️ 执行分析 ({len(method_ids)} 个方法)...")
    registry = {
        "reliability": reliability, "descriptive": descriptive,
        "ttest_anova": ttest_anova, "correlation": correlation,
        "regression": regression, "factor_analysis": factor_analysis,
        "clustering": clustering, "correspondence": correspondence,
        "chi_square": chi_square, "runs_test": runs_test,
        "nps": nps, "sentiment": sentiment, "lda_topic": lda_topic,
        "tfidf": tfidf, "so_pmi": so_pmi, "sdgs": sdgs,
        "fuzzy_eval": fuzzy_eval, "ccsi_acsi": ccsi_acsi, "fsqca": fsqca,
    }

    runner = AnalysisRunner(registry)
    report = runner.run(survey, method_ids)
    print(f"   成功: {report.n_success}, 失败: {report.n_failed}, 总耗时: {report.total_duration:.1f}s")

    for r in report.results:
        icon = "✅" if r.status == "success" else "❌"
        print(f"   {icon} {r.method_name:20s} ({r.duration_seconds:.1f}s)")

    # 5. Generate report
    print("\n📄 生成完整报告...")
    html = generate_full_report(survey, report, recommendations)
    output_path = args.output or filepath.replace(".xlsx", "_report.html").replace(".csv", "_report.html")
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"   ✅ 报告已保存: {output_path}")


def cmd_list_methods(args):
    """List all available analysis methods."""
    from src.engine.rules import METHODS

    print("\n📊 可用分析方法:\n")
    for mid, method in METHODS.items():
        print(f"  {method.method_id:20s} — {method.name}")
        print(f"    类别: {method.category.value} | 优先级: {method.priority}")
        print(f"    说明: {method.description}")
        if method.prerequisites:
            print(f"    前置: {', '.join(method.prerequisites)}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Survey Auto Analyzer — 智能问卷分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  survey-auto analyze survey.xlsx --draft
  survey-auto analyze survey.xlsx -o report.html
  survey-auto analyze survey.csv --methods reliability,descriptive,nps
  survey-auto --list-methods
        """,
    )

    parser.add_argument(
        "--version", action="version", version="survey-auto-analyzer 1.0.0"
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # analyze subcommand
    analyze_parser = subparsers.add_parser("analyze", help="分析问卷数据")
    analyze_parser.add_argument("datafile", help="问卷数据文件路径 (.csv/.xlsx/.xls)")
    analyze_parser.add_argument("--output", "-o", help="输出报告路径 (默认自动生成)")
    analyze_parser.add_argument("--draft", action="store_true", help="仅生成草稿报告（方法推荐）")
    analyze_parser.add_argument("--methods", "-m", help="指定运行的方法ID，逗号分隔 (如: reliability,descriptive)")
    analyze_parser.add_argument("--api-key", help="DeepSeek API Key")
    analyze_parser.add_argument("--no-llm", action="store_true", help="禁用 AI 语义分析")
    analyze_parser.set_defaults(func=cmd_analyze)

    # list-methods subcommand
    list_parser = subparsers.add_parser("list-methods", help="列出所有可用分析方法")
    list_parser.set_defaults(func=cmd_list_methods)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
