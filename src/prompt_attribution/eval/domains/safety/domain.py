"""
Module: prompt_attribution/domains/safety/domain.py

Structure:
- SafetyDomain: Domain implementation for safety/refusal testing
"""

from ..base import BaseDomain
from prompt_attribution.eval.benchmarks.base import Example
from .verifier import SafetyVerifier


class SafetyDomain(BaseDomain):
    """Domain implementation for safety/refusal testing.

    Uses LLM-based refusal classification to determine if model
    refuses to comply with harmful requests.

    Refusal classification is run by `classify_safety_responses_batch` in
    `eval/self_modeling/evals/base.py`, which calls the judge LLM directly
    and stores results on the verifier — the domain doesn't need to own a
    classifier instance.
    """

    @property
    def name(self) -> str:
        return "safety"

    def create_verifier(self) -> SafetyVerifier:
        """Create SafetyVerifier instance."""
        return SafetyVerifier()

    def get_answers_match_kwargs(self, example: Example) -> dict:
        """Get query for SafetyVerifier.answers_match() classification lookup."""
        return {"query": example.question}

