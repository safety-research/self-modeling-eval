"""
Module: prompt_attribution/domains/fairness/domain.py

Structure:
- FairnessDomain: Domain implementation for fairness/bias testing (BBQ)
"""


from ..base import BaseDomain
from prompt_attribution.eval.benchmarks.base import Example
from .verifier import FairnessVerifier


class FairnessDomain(BaseDomain):
    """Domain implementation for fairness/bias testing with multiple-choice questions.

    Uses FairnessVerifier for answer option comparison.
    Works with BBQ (Bias Benchmark for QA) and similar MCQ benchmarks.
    """

    @property
    def name(self) -> str:
        return "fairness"

    def create_verifier(self) -> FairnessVerifier:
        """Create FairnessVerifier instance."""
        return FairnessVerifier()

    def get_answers_match_kwargs(self, example: Example) -> dict:
        """Get additional kwargs for verifier.answers_match().

        For fairness domain, we don't need additional kwargs.

        Args:
            example: The example being processed

        Returns:
            Empty dict (no additional kwargs needed)
        """
        return {}

