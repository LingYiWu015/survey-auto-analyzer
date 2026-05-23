"""Configuration for Survey Auto Analyzer.

Reads API keys and settings from environment variables or Streamlit secrets.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Application configuration."""

    # DeepSeek API
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
    )
    deepseek_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    )
    deepseek_max_tokens: int = 4096
    deepseek_temperature: float = 0.3

    # Analysis defaults
    max_missing_rate: float = 0.3
    min_sample_size: int = 30
    significance_level: float = 0.05

    # Likert detection patterns
    likert_patterns: tuple = (
        "非常不满意", "不满意", "一般", "满意", "非常满意",
        "非常不同意", "不同意", "中立", "同意", "非常同意",
        "从不", "很少", "有时", "经常", "总是",
        "非常低", "较低", "中等", "较高", "非常高",
        "1-非常不满意", "2-不满意", "3-一般", "4-满意", "5-非常满意",
    )

    # NPS detection
    nps_patterns: tuple = (
        "推荐", "nps", "净推荐值", "0-10", "0-10分",
        "recommend", "promoter",
    )

    def load_streamlit_secrets(self):
        """Override config from Streamlit secrets if available."""
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "deepseek" in st.secrets:
                ds = st.secrets["deepseek"]
                self.deepseek_api_key = ds.get("api_key", self.deepseek_api_key)
                self.deepseek_base_url = ds.get("base_url", self.deepseek_base_url)
                self.deepseek_model = ds.get("model", self.deepseek_model)
        except ImportError:
            pass


# Singleton
config = Config()
