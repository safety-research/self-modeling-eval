"""
Module: prompt_attribution/benchmarks/base.py

Structure:
- Example: Base dataclass for benchmark examples
- BaseBenchmark: Abstract base class for benchmark loaders
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Example:
    """Base class for benchmark examples."""
    idx: int
    question: str  # The problem/question text

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"idx": self.idx, "question": self.question}


class BaseBenchmark(ABC):
    """Abstract base class for benchmark loaders."""

    @property
    @abstractmethod
    def benchmark_id(self) -> str:
        """Unique identifier for this benchmark."""
        pass

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain this benchmark belongs to (math, coding, etc)."""
        pass

    @abstractmethod
    def load_examples(
        self,
        n_samples: int,
        random_seed: int = 42,
    ) -> list[Example]:
        """Load examples from the benchmark.

        Args:
            n_samples: Number of examples to load
            random_seed: Random seed for reproducibility

        Returns:
            List of Example objects
        """
        pass

    @abstractmethod
    def make_baseline_prompt(self, example: Example, instruction: str = "") -> str:
        """Create baseline prompt for an example.

        Args:
            example: The example to create prompt for
            instruction: Optional instruction to append

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def make_lever_prompt(
        self,
        example: Example,
        lever_instruction: str,
        baseline_instruction: str = "",
    ) -> str:
        """Create lever prompt for an example.

        Args:
            example: The example to create prompt for
            lever_instruction: The lever instruction to add
            baseline_instruction: Optional baseline instruction

        Returns:
            Formatted prompt string
        """
        pass

    def get_problem_for_attribution(self, example: Example) -> str:
        """Get the problem text formatted for display in the self-modeling prompt.

        This returns the problem WITH benchmark-specific framing (e.g., "Complete
        the following Python function:", format instructions) but WITHOUT the
        lever/baseline instruction. The attribution template will add the
        instruction separately.

        Override this method in subclasses to include benchmark-specific formatting.

        Args:
            example: The example

        Returns:
            Problem text with benchmark-specific framing
        """
        # Default implementation: just return the question
        return example.question
