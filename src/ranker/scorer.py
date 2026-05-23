"""Multi-dimensional scorer for ranking analysis methods.

Scores each recommended method on:
1. Data fit (0-40): Sample size, variable type match, missing rate
2. Semantic relevance (0-30): From LLM analysis
3. Method priority (0-20): Base priority from method registry
4. Dependency bonus (0-10): Prerequisites satisfied → bonus
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScoredMethod:
    """A method with its multi-dimensional score breakdown."""

    method_id: str
    name: str
    category: str
    description: str

    # Score components
    score_data_fit: float = 0       # 0-40
    score_semantic: float = 0       # 0-30
    score_priority: float = 0       # 0-20
    score_dependency: float = 0     # 0-10
    total_score: float = 0          # 0-100

    # Metadata
    trigger_reasons: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    is_foundational: bool = False
    llm_boost: int = 0


def score_methods(
    recommendations,
    survey_ctx,
    llm_suggestions: Optional[List[Dict]] = None,
) -> List[ScoredMethod]:
    """Score and rank analysis method recommendations.

    Args:
        recommendations: List of MethodRecommendation from rule_engine.
        survey_ctx: SurveyContext from rules.
        llm_suggestions: Optional LLM method suggestions.

    Returns:
        List of ScoredMethod sorted by total_score descending.
    """
    scored = []

    # Build LLM lookup
    llm_map: Dict[str, Dict] = {}
    if llm_suggestions:
        for s in llm_suggestions:
            mid = s.get("method_id", "")
            if mid:
                llm_map[mid] = s

    for rec in recommendations:
        # 1. Data fit score (0-40)
        score_data = _compute_data_fit(rec.method_id, survey_ctx)

        # 2. Semantic relevance (0-30) — from LLM only
        llm_info = llm_map.get(rec.method_id, {})
        score_semantic = min(30, llm_info.get("extra_score", 0))

        # 3. Method priority (0-20) — normalize from base_priority (0-100)
        score_priority = rec.base_priority / 100 * 20

        # 4. Dependency bonus (0-10)
        score_dep = _compute_dependency_bonus(rec, recommendations)

        total = score_data + score_semantic + score_priority + score_dep

        scored.append(ScoredMethod(
            method_id=rec.method_id,
            name=rec.name,
            category=rec.category,
            description=rec.description,
            score_data_fit=score_data,
            score_semantic=score_semantic,
            score_priority=score_priority,
            score_dependency=score_dep,
            total_score=round(total, 1),
            trigger_reasons=rec.trigger_reasons,
            prerequisites=rec.prerequisites,
            is_foundational=rec.is_foundational,
            llm_boost=int(score_semantic),
        ))

    # Sort by total_score descending
    scored.sort(key=lambda s: (s.is_foundational, s.total_score), reverse=True)
    return scored


def _compute_data_fit(method_id: str, ctx) -> float:
    """Compute data fit score (0-40) based on survey characteristics."""
    score = 20.0  # Start at midpoint

    # Sample size bonus
    if ctx.sample_size >= 200:
        score += 10
    elif ctx.sample_size >= 100:
        score += 6
    elif ctx.sample_size >= 50:
        score += 3
    elif ctx.sample_size >= 30:
        score += 0
    else:
        score -= 10  # Small sample penalty

    # Variable type match
    type_requirements = {
        "reliability": ctx.n_likert,
        "factor_analysis": ctx.n_likert,
        "ttest_anova": ctx.n_categorical + ctx.n_numeric,
        "correlation": ctx.n_numeric,
        "regression": ctx.n_numeric,
        "clustering": ctx.n_numeric,
        "chi_square": ctx.n_categorical,
        "correspondence": ctx.n_categorical,
        "nps": ctx.n_nps,
        "sentiment": ctx.n_text,
        "lda_topic": ctx.n_text,
        "tfidf": ctx.n_text,
        "so_pmi": ctx.n_text,
        "sdgs": ctx.n_text,
        "fuzzy_eval": ctx.n_likert,
        "ccsi_acsi": ctx.n_likert,
        "fsqca": ctx.n_categorical + ctx.n_likert,
    }

    req = type_requirements.get(method_id, 1)
    if req >= 10:
        score += 10
    elif req >= 5:
        score += 6
    elif req >= 3:
        score += 3
    elif req >= 1:
        score += 0
    else:
        score -= 5

    # Clamp
    return max(0, min(40, score))


def _compute_dependency_bonus(rec, all_recs) -> float:
    """Compute dependency bonus (0-10).

    If all prerequisites are also recommended → bonus.
    If this method is a prerequisite for others → small bonus.
    """
    score = 0.0

    # Check if prerequisites are met
    prereqs = set(rec.prerequisites)
    recommended_ids = {r.method_id for r in all_recs}

    if prereqs:
        satisfied = prereqs & recommended_ids
        score += len(satisfied) / len(prereqs) * 5  # Up to 5 points

    # Bonus if this method enables others
    enables_count = sum(
        1 for r in all_recs if rec.method_id in r.prerequisites
    )
    score += min(5, enables_count * 2)  # Up to 5 points

    return min(10, score)


def get_top_methods(scored: List[ScoredMethod], n: int = 5) -> List[ScoredMethod]:
    """Get the top N scored methods."""
    return scored[:n]


def get_high_confidence_methods(
    scored: List[ScoredMethod], threshold: float = 60.0
) -> List[ScoredMethod]:
    """Get methods with total_score above threshold."""
    return [s for s in scored if s.total_score >= threshold]
