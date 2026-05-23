"""Method dependency resolver.

Defines the execution DAG for analysis methods:
- Some methods depend on others (e.g., factor_analysis needs reliability)
- Groups methods into stages for parallel execution
"""

from typing import List, Dict, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStage:
    """A group of methods that can run in parallel."""

    stage_id: int
    methods: List[str]  # Method IDs
    description: str = ""


# ─── Dependency Graph ────────────────────────────────────────────────

# For each method, list the method IDs it depends on (must complete first)
DEPENDENCY_GRAPH: Dict[str, List[str]] = {
    "reliability": [],           # No dependencies
    "descriptive": [],           # No dependencies
    "ttest_anova": [],           # No dependencies
    "correlation": [],           # No dependencies
    "chi_square": [],            # No dependencies
    "runs_test": [],             # No dependencies
    "nps": [],                   # No dependencies
    "sentiment": [],             # No dependencies
    "tfidf": [],                 # No dependencies
    "so_pmi": [],                # No dependencies
    "sdgs": [],                  # No dependencies

    "regression": ["correlation"],          # Should check correlations first
    "factor_analysis": ["reliability"],     # Need reliability test first
    "clustering": [],                        # Independent
    "correspondence": ["chi_square"],        # Chi-square before correspondence
    "lda_topic": ["sentiment"],              # Sentiment before topic modeling
    "fuzzy_eval": [],                        # Independent
    "ccsi_acsi": ["reliability", "factor_analysis"],  # Full pipeline
    "fsqca": [],                             # Independent
}

# Methods that are foundational and should always run first
FOUNDATIONAL_METHODS = ["descriptive", "reliability"]


def resolve_stages(method_ids: List[str]) -> List[ExecutionStage]:
    """Resolve method execution order into dependency stages.

    Uses topological sort with the dependency graph.
    Methods within the same stage can run in parallel.

    Args:
        method_ids: List of method IDs to execute.

    Returns:
        List of ExecutionStage objects in execution order.
    """
    # Build subgraph for requested methods only
    subgraph: Dict[str, Set[str]] = {}
    all_needed = set(method_ids)

    # Expand to include all dependencies
    changed = True
    while changed:
        changed = False
        for mid in list(all_needed):
            for dep in DEPENDENCY_GRAPH.get(mid, []):
                if dep not in all_needed:
                    all_needed.add(dep)
                    changed = True

    # Build adjacency
    for mid in all_needed:
        subgraph[mid] = set(DEPENDENCY_GRAPH.get(mid, []))

    # Topological sort into stages (Kahn's algorithm variant)
    completed: Set[str] = set()
    remaining = set(all_needed)
    stages: List[ExecutionStage] = []
    stage_num = 0

    while remaining:
        stage_num += 1
        # Find all methods whose dependencies are all completed
        ready = []
        for mid in sorted(remaining):
            deps = subgraph.get(mid, set())
            if deps.issubset(completed):
                ready.append(mid)

        if not ready:
            # Should not happen with acyclic graph, but handle gracefully
            logger.warning(f"Circular dependency detected in: {remaining}")
            ready = sorted(remaining)

        stages.append(ExecutionStage(
            stage_id=stage_num,
            methods=ready,
            description=f"Stage {stage_num}: {', '.join(ready)}",
        ))

        completed.update(ready)
        remaining.difference_update(ready)

    return stages


def get_method_dependencies(method_id: str) -> List[str]:
    """Get all direct and transitive dependencies for a method."""
    deps = set()
    _collect_deps(method_id, deps)
    return sorted(deps)


def _collect_deps(method_id: str, collected: Set[str]):
    """Recursively collect dependencies."""
    for dep in DEPENDENCY_GRAPH.get(method_id, []):
        if dep not in collected:
            collected.add(dep)
            _collect_deps(dep, collected)


def get_dependents(method_id: str) -> List[str]:
    """Get methods that depend on this method."""
    dependents = []
    for mid, deps in DEPENDENCY_GRAPH.items():
        if method_id in deps:
            dependents.append(mid)
    return dependents


def validate_plan(method_ids: List[str]) -> List[str]:
    """Validate that all dependencies are included in the plan.

    Returns list of warnings.
    """
    warnings = []
    all_ids = set(method_ids)
    for mid in method_ids:
        for dep in DEPENDENCY_GRAPH.get(mid, []):
            if dep not in all_ids:
                warnings.append(
                    f"方法 '{mid}' 依赖 '{dep}'，但 '{dep}' 未包含在分析计划中"
                )
    return warnings
