"""Rule Engine - evaluates survey characteristics against rule set.

Produces a ranked list of applicable analysis methods with trigger reasons.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging

from .rules import (
    Rule, MethodInfo, SurveyContext,
    RULES, METHODS, get_method_info,
)

logger = logging.getLogger(__name__)


@dataclass
class MethodRecommendation:
    """A recommended analysis method with metadata."""

    method_id: str
    name: str
    category: str
    description: str
    trigger_reasons: List[str]      # Which rules triggered this
    base_priority: int              # From MethodInfo
    boost: int                      # From rules
    final_priority: int             # base + boost
    prerequisites: List[str]        # Method IDs that must run first
    is_foundational: bool


def evaluate(ctx: SurveyContext) -> List[MethodRecommendation]:
    """Evaluate all rules against the survey context.

    Returns list of MethodRecommendation sorted by final_priority descending.
    """
    # Collect triggered methods with their reasons and boosts
    triggered: Dict[str, Dict] = {}  # method_id -> {reasons, total_boost}

    for rule in RULES:
        try:
            if rule.condition(ctx):
                for method_id in rule.methods:
                    if method_id not in triggered:
                        triggered[method_id] = {
                            "reasons": [],
                            "total_boost": 0,
                        }
                    triggered[method_id]["reasons"].append(rule.note)
                    triggered[method_id]["total_boost"] += rule.priority_boost
        except Exception as e:
            logger.warning(f"Rule {rule.rule_id} ({rule.name}) evaluation failed: {e}")

    # Build recommendations
    recommendations = []
    for method_id, info in triggered.items():
        method = get_method_info(method_id)
        if method is None:
            logger.warning(f"Unknown method ID: {method_id}")
            continue

        final_priority = min(100, max(0, method.priority + info["total_boost"]))

        recommendations.append(MethodRecommendation(
            method_id=method_id,
            name=method.name,
            category=method.category.value,
            description=method.description,
            trigger_reasons=info["reasons"],
            base_priority=method.priority,
            boost=info["total_boost"],
            final_priority=final_priority,
            prerequisites=method.prerequisites,
            is_foundational=method.is_foundational,
        ))

    # Sort by priority descending, foundational first
    recommendations.sort(key=lambda r: (r.is_foundational, r.final_priority), reverse=True)

    return recommendations


def get_analysis_plan(
    survey_def,
    llm_suggestions: Optional[List[Dict]] = None,
) -> List[MethodRecommendation]:
    """Generate a complete analysis plan for a survey.

    This is the main entry point for the rule engine.

    Args:
        survey_def: SurveyDefinition from parser.
        llm_suggestions: Optional LLM semantic suggestions to boost scores.

    Returns:
        Ranked list of MethodRecommendation.
    """
    ctx = SurveyContext(survey_def)
    recommendations = evaluate(ctx)

    # If LLM suggestions provided, apply additional boosts
    if llm_suggestions:
        recommendations = _apply_llm_boosts(recommendations, llm_suggestions)

    return recommendations


def _apply_llm_boosts(
    recommendations: List[MethodRecommendation],
    llm_suggestions: List[Dict],
) -> List[MethodRecommendation]:
    """Apply LLM semantic analysis boosts to recommendations."""
    # Build lookup of method_id -> boost from LLM
    llm_boosts: Dict[str, int] = {}
    for s in llm_suggestions:
        mid = s.get("method_id", "")
        extra = s.get("extra_score", 0)
        if mid:
            llm_boosts[mid] = llm_boosts.get(mid, 0) + extra

    for rec in recommendations:
        if rec.method_id in llm_boosts:
            rec.boost += llm_boosts[rec.method_id]
            rec.final_priority = min(100, rec.base_priority + rec.boost)
            rec.trigger_reasons.append(
                f"🤖 AI语义分析: +{llm_boosts[rec.method_id]}分"
            )

    # Re-sort
    recommendations.sort(key=lambda r: (r.is_foundational, r.final_priority), reverse=True)
    return recommendations


def get_dependency_order(
    recommendations: List[MethodRecommendation],
) -> List[List[str]]:
    """Group recommendations into dependency-ordered stages.

    Returns list of stages, each stage is a list of method_ids that can run in parallel.
    """
    all_ids = {r.method_id for r in recommendations}
    stages = []
    completed = set()
    remaining = set(all_ids)

    while remaining:
        stage = []
        for mid in list(remaining):
            rec = next(r for r in recommendations if r.method_id == mid)
            prereqs = set(rec.prerequisites) & all_ids
            if prereqs.issubset(completed):
                stage.append(mid)
                remaining.discard(mid)
        if not stage:
            # Circular dependency or missing prereq — run remaining sequentially
            stage = list(remaining)
            remaining.clear()
        stages.append(stage)
        completed.update(stage)

    return stages


def summarize_plan(recommendations: List[MethodRecommendation]) -> str:
    """Generate a human-readable summary of the analysis plan."""
    lines = ["## 分析计划\n"]
    lines.append(f"共推荐 {len(recommendations)} 个分析方法：\n")

    for i, rec in enumerate(recommendations, 1):
        star = "⭐" if rec.final_priority >= 80 else "📊" if rec.final_priority >= 50 else "🔍"
        lines.append(f"{i}. {star} **{rec.name}** (优先级: {rec.final_priority})")
        lines.append(f"   - 类别: {rec.category}")
        lines.append(f"   - {rec.description}")
        if rec.prerequisites:
            prereq_names = [
                get_method_info(p).name if get_method_info(p) else p
                for p in rec.prerequisites
            ]
            lines.append(f"   - 前置方法: {', '.join(prereq_names)}")
        for reason in rec.trigger_reasons:
            lines.append(f"   - 触发原因: {reason}")
        lines.append("")

    return "\n".join(lines)
