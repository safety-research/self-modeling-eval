"""
Module: prompt_attribution/domains/coding/domain.py

Structure:
- CodingDomain: Domain implementation for coding problems
"""

from typing import Optional

from ..base import BaseDomain
from prompt_attribution.eval.benchmarks.base import Example
from prompt_attribution.shared.config import PerturbationConfig
from .verifier import CodeVerifier


class CodingDomain(BaseDomain):
    """Domain implementation for coding problems.

    Uses CodeVerifier for AST-based feature comparison.
    """

    def __init__(self, perturbation_config: PerturbationConfig):
        super().__init__(perturbation_config)
        self._verifier: Optional[CodeVerifier] = None

    @property
    def name(self) -> str:
        return "coding"

    def create_verifier(self) -> CodeVerifier:
        """Create CodeVerifier with target features from config."""
        self._verifier = CodeVerifier(
            target_features=self.perturbation_config.target_features
        )
        return self._verifier

    def get_answers_match_kwargs(self, example: Example) -> dict:
        """Get kwargs for CodeVerifier.answers_match().

        For code examples, returns entry_point and function_prompt.
        """

        return {
            "entry_point": getattr(example, "entry_point", ""),
            "function_prompt": getattr(example, "prompt", ""),
        }

