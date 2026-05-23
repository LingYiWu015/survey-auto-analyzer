"""DeepSeek API client wrapper.

Uses the OpenAI-compatible API format. DeepSeek's API is fully compatible
with the OpenAI Python SDK — just change the base_url.
"""

import json
import logging
from typing import Optional, Dict, List, Any

from openai import OpenAI

from src.config import config

logger = logging.getLogger(__name__)

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds


class DeepSeekClient:
    """Wrapper around DeepSeek API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or config.deepseek_api_key
        self.base_url = base_url or config.deepseek_base_url
        self.model = model or config.deepseek_model

        if not self.api_key:
            logger.warning(
                "DeepSeek API key not configured. "
                "Set DEEPSEEK_API_KEY environment variable or configure in Streamlit secrets."
            )
            self._client = None
        else:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

    @property
    def is_available(self) -> bool:
        return self._client is not None and bool(self.api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> Optional[str]:
        """Send a chat completion request.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            json_mode: If True, request JSON output format.

        Returns:
            Response text, or None on failure.
        """
        if not self.is_available:
            logger.error("DeepSeek client not available (no API key).")
            return None

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                return content
            except Exception as e:
                logger.warning(
                    f"DeepSeek API attempt {attempt + 1}/{MAX_RETRIES} failed: {e}"
                )
                if attempt < MAX_RETRIES - 1:
                    import time
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"DeepSeek API call failed after {MAX_RETRIES} retries")
                    return None

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Optional[Dict[str, Any]]:
        """Send a chat request and parse response as JSON.

        Returns parsed JSON dict, or None on failure.
        """
        text = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        if text is None:
            return None

        try:
            # Strip markdown code fences if present
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {text[:500]}")
            return {"raw_response": text, "parse_error": str(e)}

    def analyze_survey_semantics(
        self,
        questions: List[Dict[str, str]],
    ) -> Optional[Dict[str, Any]]:
        """Analyze survey question semantics using DeepSeek.

        Args:
            questions: List of {col_name, var_type, label, options} dicts.

        Returns:
            Dict with 'questions', 'hypotheses', 'method_suggestions'.
        """
        from .prompts import SEMANTIC_ANALYSIS_PROMPT

        questions_text = _format_questions_for_prompt(questions)
        prompt = SEMANTIC_ANALYSIS_PROMPT.format(questions=questions_text)

        messages = [
            {"role": "system", "content": "你是一位资深的市场调查和数据分析专家，精通定量研究、文本分析和综合评价方法。请严格按JSON格式输出。"},
            {"role": "user", "content": prompt},
        ]

        return self.chat_json(messages, temperature=0.3)


# Global singleton
_client: Optional[DeepSeekClient] = None


def get_client(force_refresh: bool = False) -> DeepSeekClient:
    """Get or create the global DeepSeek client.

    Detects API key changes and rebuilds the client automatically.
    Set force_refresh=True to force re-creation even if key unchanged.
    """
    global _client, _last_api_key
    current_key = config.deepseek_api_key
    if _client is None or force_refresh or _last_api_key != current_key:
        _client = DeepSeekClient()
        _last_api_key = current_key
    return _client


# Track last used API key for hot-reload detection
_last_api_key: str = ""


def _format_questions_for_prompt(questions: List[Dict[str, str]]) -> str:
    """Format question list into a prompt-friendly string."""
    lines = []
    for i, q in enumerate(questions, 1):
        lines.append(
            f"{i}. [{q.get('var_type', 'unknown')}] "
            f"{q.get('label', q.get('col_name', ''))}"
        )
        opts = q.get("options", [])
        if opts and len(opts) <= 10:
            lines.append(f"   选项: {', '.join(opts[:10])}")
    return "\n".join(lines)
