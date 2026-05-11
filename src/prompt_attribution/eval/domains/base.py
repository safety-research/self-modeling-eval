"""
Module: prompt_attribution/domains/base.py

Structure:
- BaseVerifier: Abstract base class for all verifiers
- BaseDomain: Abstract base class for all domains

Design principle: Domain-specific logic should live in domain classes,
not in runners. Runners call domain methods to handle domain-specific
behavior without hardcoding domain names.
"""

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from prompt_attribution.eval.benchmarks.base import Example
from prompt_attribution.shared.config import PerturbationConfig

if TYPE_CHECKING:
    pass


class BaseVerifier(ABC):
    """Abstract base class for answer verification."""

    @abstractmethod
    def parse_answer(self, raw_output: str) -> Any:
        """Parse answer from model output.

        Args:
            raw_output: Full model response text

        Returns:
            Parsed answer (type depends on domain)
        """
        pass

    @abstractmethod
    def answers_match(self, answer1: Any, answer2: Any) -> bool:
        """Compare two answers for equivalence.

        Args:
            answer1: First parsed answer
            answer2: Second parsed answer

        Returns:
            True if answers are equivalent
        """
        pass


class BaseDomain(ABC):
    """Abstract base class for domain-specific logic.

    Each domain (math, coding, safety, etc.) implements this interface
    to handle domain-specific operations like verifier creation,
    domain data extraction, and YES/NO explanation building.
    """

    def __init__(self, perturbation_config: PerturbationConfig):
        """Initialize domain with perturbation config.

        Args:
            perturbation_config: Configuration for the perturbation being tested
        """
        self.perturbation_config = perturbation_config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return domain name (e.g., 'math', 'coding')."""
        pass

    @abstractmethod
    def create_verifier(self):
        """Create domain-appropriate verifier.

        Returns:
            Verifier instance for this domain
        """
        pass

    def get_answers_match_kwargs(self, example: Example) -> dict:
        """Get extra kwargs for verifier.answers_match() call.

        Override in domains that need to pass additional arguments
        to answers_match (e.g., coding needs entry_point/function_prompt,
        safety needs query for classification lookup).

        Args:
            example: The example being processed

        Returns:
            Dict of extra kwargs to pass to verifier.answers_match()
        """
        return {}

