"""Analysis Execution Runner.

Schedules and executes analysis methods in dependency order.
Each method is isolated — a failure in one doesn't block others.
"""

import traceback
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..ranker.dependency import resolve_stages, validate_plan

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result from a single analysis method execution."""

    method_id: str
    method_name: str
    status: str  # "success", "failed", "skipped", "not_applicable"
    duration_seconds: float = 0.0
    error_message: str = ""

    # Analysis outputs
    tables: List[Dict[str, Any]] = field(default_factory=list)
    charts: List[Dict[str, Any]] = field(default_factory=list)
    interpretation: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_id": self.method_id,
            "method_name": self.method_name,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "tables": self.tables,
            "charts": self.charts,
            "interpretation": self.interpretation,
            "warnings": self.warnings,
        }


@dataclass
class ExecutionReport:
    """Complete execution report for all analyses."""

    survey_name: str = ""
    executed_at: str = ""
    total_duration: float = 0.0
    results: List[AnalysisResult] = field(default_factory=list)
    n_success: int = 0
    n_failed: int = 0
    n_skipped: int = 0

    @property
    def success_rate(self) -> float:
        total = self.n_success + self.n_failed
        return self.n_success / max(total, 1) * 100

    def get_result(self, method_id: str) -> Optional[AnalysisResult]:
        for r in self.results:
            if r.method_id == method_id:
                return r
        return None


class AnalysisRunner:
    """Executes analysis methods in dependency-respecting order."""

    def __init__(self, method_registry: Dict[str, Callable]):
        """Initialize with a registry of method functions.

        Args:
            method_registry: Dict mapping method_id -> callable(survey_def, **params).
        """
        self.registry = method_registry

    def run(
        self,
        survey_def,
        method_ids: List[str],
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> ExecutionReport:
        """Execute a list of analysis methods.

        Args:
            survey_def: SurveyDefinition from parser.
            method_ids: List of method IDs to execute.
            progress_callback: Optional callback(status, current, total) for progress.
            **kwargs: Additional params passed to each method.

        Returns:
            ExecutionReport with all results.
        """
        # Validate plan
        warnings = validate_plan(method_ids)
        if warnings:
            for w in warnings:
                logger.warning(w)

        # Resolve execution stages
        stages = resolve_stages(method_ids)

        report = ExecutionReport(
            survey_name=getattr(survey_def, "filepath", "Unknown"),
            executed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        total_methods = sum(len(s.methods) for s in stages)
        completed = 0
        start_time = datetime.now()

        for stage in stages:
            logger.info(f"Executing stage {stage.stage_id}: {stage.methods}")

            for method_id in stage.methods:
                completed += 1

                if progress_callback:
                    progress_callback(
                        "running", method_id, completed, total_methods
                    )

                result = self._run_one(survey_def, method_id, **kwargs)
                report.results.append(result)

                if result.status == "success":
                    report.n_success += 1
                elif result.status == "failed":
                    report.n_failed += 1
                else:
                    report.n_skipped += 1

        report.total_duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Execution complete: {report.n_success} success, "
            f"{report.n_failed} failed, {report.n_skipped} skipped "
            f"({report.total_duration:.1f}s)"
        )

        return report

    def _run_one(self, survey_def, method_id: str, **kwargs) -> AnalysisResult:
        """Execute a single analysis method with error isolation."""
        import time

        from ..engine.rules import get_method_info

        method_info = get_method_info(method_id)
        method_name = method_info.name if method_info else method_id

        func = self.registry.get(method_id)
        if func is None:
            return AnalysisResult(
                method_id=method_id,
                method_name=method_name,
                status="skipped",
                error_message=f"Method '{method_id}' not found in registry",
            )

        t0 = time.time()
        try:
            # Check applicability first
            if hasattr(func, "check_applicability"):
                applicable, reason = func.check_applicability(survey_def)
                if not applicable:
                    return AnalysisResult(
                        method_id=method_id,
                        method_name=method_name,
                        status="not_applicable",
                        error_message=reason,
                    )

            # Run analysis
            output = func.run(survey_def, **kwargs)
            duration = round(time.time() - t0, 2)

            return AnalysisResult(
                method_id=method_id,
                method_name=method_name,
                status="success",
                duration_seconds=duration,
                tables=output.get("tables", []),
                charts=output.get("charts", []),
                interpretation=output.get("interpretation", ""),
                warnings=output.get("warnings", []),
            )

        except Exception as e:
            duration = round(time.time() - t0, 2)
            logger.error(f"Method '{method_id}' failed: {traceback.format_exc()}")
            return AnalysisResult(
                method_id=method_id,
                method_name=method_name,
                status="failed",
                duration_seconds=duration,
                error_message=str(e),
            )
