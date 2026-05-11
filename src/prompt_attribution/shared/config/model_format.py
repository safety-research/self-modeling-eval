"""
Module: prompt_attribution/shared/config/model_format.py

`ModelFormat` auto-detects thinking mode from a model id and builds
provider-specific request kwargs for vLLM / Anthropic / OpenAI / Gemini /
Together. The self-modeling runner uses it as the single source of truth
for "should this model think, and how do we tell its provider so?".
"""

from dataclasses import dataclass


@dataclass
class ModelFormat:
    """Controls completion format and output parsing per model family.

    Thinking models (Qwen3, DeepSeek-R1) use <think>...</think> tags.
    Non-thinking models (Llama, Mistral) put reasoning in the JSON field.

    Set enable_thinking=False on a thinking-capable model (e.g., Qwen3) to
    force non-thinking mode for both training and inference, avoiding
    distribution shift between train and eval.
    """

    thinking: bool = False
    enable_thinking: bool | None = None  # None = auto (True for thinking models)
    thinking_budget: int = 0  # 0 = no thinking; >0 = cap thinking tokens
    reasoning_parser: str = ""  # vLLM --reasoning-parser flag (e.g., "qwen3", "kimi_k2")

    def get_thinking_extra_body(self) -> dict:
        """Build extra_body dict for vLLM thinking budget.

        Returns empty dict for non-thinking models (no extra_body needed).
        For thinking-capable models, explicitly sets budget (on or off).

        NOTE: This is for vLLM only (requires vLLM >= 0.19.0 for enforcement).
        For cloud API providers, use get_api_thinking_kwargs(provider) instead.
        """
        if not self.thinking:
            return {}
        if self.enable_thinking is False:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        if self.thinking_budget == 0:
            # Thinking via system prompt, not extra_body (e.g., GPT-OSS)
            return {}
        # thinking_token_budget: vLLM >= 0.19 hard-caps thinking tokens
        return {"thinking_token_budget": self.thinking_budget}

    def get_api_thinking_kwargs(self, provider: str, model_name: str = "") -> dict:
        """Build provider-specific kwargs for cloud API thinking.

        - anthropic: thinking={"type":"enabled","budget_tokens":N}, temperature=1
        - together: reasoning={"enabled":True} + optional thinking_budget
        - openai: reasoning_effort="high" for reasoning models (o-series, gpt-5+)
        - google/gemini: thinking_budget=N (handled by our safetytooling patch)
        """
        if not self.thinking or self.enable_thinking is False:
            return {}

        if provider == "anthropic":
            budget = max(self.thinking_budget, 1024)  # Anthropic min is 1024
            return {
                "thinking": {"type": "enabled", "budget_tokens": budget},
                "temperature": 1,
            }
        elif provider == "together":
            result: dict = {"reasoning": {"enabled": True}}
            if self.thinking_budget > 0:
                result["extra_body"] = {"thinking_budget": self.thinking_budget}
            return result
        elif provider == "openai":
            name_lower = model_name.lower()
            is_reasoning = any(k in name_lower for k in ("o1-", "o3", "o4", "gpt-5"))
            if is_reasoning:
                return {"reasoning_effort": "high", "temperature": 1}
            return {}
        elif provider in ("google", "gemini"):
            if self.thinking_budget > 0:
                return {"thinking_budget": self.thinking_budget}
            return {}
        else:
            return {}

    @classmethod
    def from_model_name(cls, model_name: str, max_tokens: int = 2048) -> "ModelFormat":
        """Auto-detect format from model name.

        Thinking budget is half of max_tokens — thinking and content share the
        same vLLM token budget, so reserving half for each prevents thinking
        from starving content. GPT-OSS uses system prompt instead of
        extra_body (handled separately).
        """
        name_lower = model_name.lower()
        thinking_budget = max_tokens // 2
        if "qwen3" in name_lower:
            return cls(
                thinking=True, enable_thinking=True,
                thinking_budget=thinking_budget, reasoning_parser="qwen3",
            )
        if "deepseek" in name_lower:
            return cls(
                thinking=True, enable_thinking=True,
                thinking_budget=thinking_budget, reasoning_parser="deepseek_r1",
            )
        if "kimi" in name_lower:
            return cls(
                thinking=True, enable_thinking=True,
                thinking_budget=thinking_budget, reasoning_parser="kimi_k2",
            )
        if "gpt-oss" in name_lower:
            return cls(
                thinking=True, enable_thinking=True,
                thinking_budget=0, reasoning_parser="openai_gptoss",
            )
        return cls(thinking=False, enable_thinking=False)
