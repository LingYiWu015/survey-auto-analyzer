"""Survey Parser - orchestrates data loading and variable classification.

Produces a structured SurveyDefinition that describes:
- Each question's text, type, options, and sample values
- Overall survey metadata (sample size, platform, etc.)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import pandas as pd
import logging

from .data_loader import load_data
from .variable_classifier import classify_all_columns, get_variable_summary

logger = logging.getLogger(__name__)


@dataclass
class QuestionDef:
    """Definition of a single survey question."""

    col_name: str              # Original column name / question text
    var_type: str              # single_choice, multi_choice, likert_scale, etc.
    label: str                 # Cleaned question text for display
    options: List[str] = field(default_factory=list)  # Unique answer options
    n_unique: int = 0
    n_missing: int = 0
    missing_rate: float = 0.0
    sample_values: List[str] = field(default_factory=list)
    # Metadata for Likert detection
    is_reverse_scored: bool = False


@dataclass
class SurveyDefinition:
    """Complete survey structure definition."""

    filepath: str = ""
    platform: str = "generic"
    sample_size: int = 0
    n_questions: int = 0
    questions: List[QuestionDef] = field(default_factory=list)
    var_types_summary: Dict[str, int] = field(default_factory=dict)
    # Raw data reference (not serialized)
    _data: Optional[pd.DataFrame] = None

    @property
    def df(self) -> pd.DataFrame:
        """Get the raw data DataFrame."""
        if self._data is None:
            raise ValueError("Data not loaded. Call parse_survey() first.")
        return self._data

    def get_questions_by_type(self, var_type: str) -> List[QuestionDef]:
        """Get all questions of a specific type."""
        return [q for q in self.questions if q.var_type == var_type]

    def get_question(self, col_name: str) -> Optional[QuestionDef]:
        """Get a question by column name."""
        for q in self.questions:
            if q.col_name == col_name:
                return q
        return None

    def has_type(self, var_type: str) -> bool:
        """Check if any questions of this type exist."""
        return self.var_types_summary.get(var_type, 0) > 0

    def has_text(self) -> bool:
        return self.has_type("text")

    def has_likert(self) -> bool:
        return self.has_type("likert_scale")

    def has_categorical(self) -> bool:
        return self.has_type("single_choice") or self.has_type("multi_choice")

    def has_continuous(self) -> bool:
        return self.has_type("continuous") or self.has_type("likert_scale")

    def has_nps(self) -> bool:
        return self.has_type("nps")

    def has_demographic(self) -> bool:
        return self.has_type("demographic")


def parse_survey(filepath: str, sheet_name: int = 0) -> SurveyDefinition:
    """Parse a survey data file and return a structured SurveyDefinition.

    This is the main entry point for survey parsing.
    """
    logger.info(f"Parsing survey: {filepath}")

    # Load data
    df, platform = load_data(filepath, sheet_name)

    # Classify all columns
    classification = classify_all_columns(df)

    # Build SurveyDefinition
    survey = SurveyDefinition(
        filepath=filepath,
        platform=platform,
        sample_size=len(df),
        n_questions=0,
        _data=df,
    )

    # Count types
    type_counts: Dict[str, int] = {}

    for col in df.columns:
        var_type = classification.get(col, "unknown")

        # Skip pure metadata columns (but keep demographics)
        if var_type == "metadata":
            continue

        type_counts[var_type] = type_counts.get(var_type, 0) + 1

        series = df[col]
        non_null = series.dropna()
        n_missing = series.isna().sum()

        # Get unique options
        unique_vals = non_null.unique()
        options = [str(v) for v in unique_vals[:30]]  # Cap at 30

        # Clean question label
        label = _clean_question_label(col)

        qdef = QuestionDef(
            col_name=col,
            var_type=var_type,
            label=label,
            options=options,
            n_unique=len(unique_vals),
            n_missing=int(n_missing),
            missing_rate=round(n_missing / max(len(df), 1), 4),
            sample_values=[str(v) for v in unique_vals[:5]],
        )
        survey.questions.append(qdef)

    survey.n_questions = len(survey.questions)
    survey.var_types_summary = type_counts

    logger.info(
        f"Parsed survey: {survey.n_questions} questions, "
        f"{survey.sample_size} responses, platform={platform}"
    )
    return survey


def _clean_question_label(col_name: str) -> str:
    """Clean up a question label for display.

    Removes common prefixes like 'Q1.', '1.', etc.
    """
    # Remove leading numbers and dots
    cleaned = col_name.strip()
    cleaned = cleaned.split("\n")[0]  # Take first line if multi-line

    # Truncate if too long
    if len(cleaned) > 80:
        cleaned = cleaned[:77] + "..."

    return cleaned


def get_analysis_summary(survey: SurveyDefinition) -> pd.DataFrame:
    """Generate a summary of the parsed survey for display."""
    return get_variable_summary(survey.df)
