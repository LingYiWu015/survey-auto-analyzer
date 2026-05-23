"""Quick validation test for Survey Auto Analyzer.

Tests the core pipeline without Streamlit:
1. Parse sample survey
2. Rule engine recommendations
3. Execute selected methods
4. Generate report

Usage:
    python -m tests.validate
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser.survey_parser import parse_survey
from src.engine.rule_engine import get_analysis_plan, summarize_plan
from src.engine.rules import SurveyContext
from src.ranker.scorer import score_methods
from src.executor.runner import AnalysisRunner
from src.executor.methods import (
    reliability, descriptive, ttest_anova, correlation,
    regression, factor_analysis, clustering, correspondence,
    chi_square, runs_test, nps, sentiment, lda_topic,
    tfidf, so_pmi, sdgs, fuzzy_eval, ccsi_acsi, fsqca,
)
from src.reporter.generator import generate_full_report


def main():
    # Find sample data
    sample_csv = Path(__file__).parent / "sample_data" / "sample_survey_data.csv"

    if not sample_csv.exists():
        print("Generating sample data first...")
        from tests.sample_data.generate_sample import generate_sample_data
        sample_csv = Path(generate_sample_data())

    print(f"\n{'='*60}")
    print("  Survey Auto Analyzer — Validation Test")
    print(f"{'='*60}\n")

    # 1. Parse survey
    print("[1/5] Parsing survey...")
    survey = parse_survey(str(sample_csv))
    print(f"  ✅ Loaded: {survey.sample_size} responses, {survey.n_questions} questions")
    print(f"  Types: {survey.var_types_summary}")

    # 2. Rule engine
    print("\n[2/5] Running rule engine...")
    recommendations = get_analysis_plan(survey)
    ctx = SurveyContext(survey)
    scored = score_methods(recommendations, ctx)
    print(f"  ✅ Recommended {len(scored)} methods:")
    for s in scored[:8]:
        print(f"     {s.name:20s} → {s.total_score:.0f}分 | {s.description}")

    # 3. Execute top methods
    print("\n[3/5] Executing analyses...")
    registry = {
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
    runner = AnalysisRunner(registry)

    # Select top methods
    top_methods = [s.method_id for s in scored if s.total_score >= 40]
    print(f"  Running {len(top_methods)} methods: {top_methods}")

    report = runner.run(survey, top_methods)
    print(f"  ✅ Done: {report.n_success} success, {report.n_failed} failed, {report.n_skipped} skipped")
    print(f"  Duration: {report.total_duration:.1f}s")

    # 4. Show results
    print("\n[4/5] Results:")
    for result in report.results:
        status = "✅" if result.status == "success" else "❌"
        print(f"  {status} {result.method_name:20s} ({result.duration_seconds:.1f}s)")
        if result.status == "success":
            print(f"     Tables: {len(result.tables)}, Warnings: {len(result.warnings)}")
            if result.interpretation:
                print(f"     → {result.interpretation[:100]}...")
        else:
            print(f"     → {result.error_message[:100]}")

    # 5. Generate report
    print("\n[5/5] Generating report...")
    html = generate_full_report(survey, report, recommendations)
    report_path = Path(__file__).parent / "sample_data" / "test_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"  ✅ Report saved to: {report_path}")
    print(f"  Report size: {len(html)} chars")

    print(f"\n{'='*60}")
    print("  ✅ Validation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
