"""
Module: prompt_attribution/benchmarks/humaneval.py

Structure:
- HumanEvalExample: Dataclass for HumanEval examples
- HumanEvalBenchmark: Benchmark loader for HumanEval code generation
"""

import random
from dataclasses import dataclass

from datasets import load_dataset

from .base import BaseBenchmark, Example


@dataclass
class HumanEvalExample(Example):
    """HumanEval code generation example."""
    task_id: str = ""
    prompt: str = ""  # Function signature and docstring
    canonical_solution: str = ""
    test: str = ""
    entry_point: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "idx": self.idx,
            "task_id": self.task_id,
            "question": self.question,  # Same as prompt for consistency
            "prompt": self.prompt,
            "canonical_solution": self.canonical_solution,
            "test": self.test,
            "entry_point": self.entry_point,
        }


class HumanEvalBenchmark(BaseBenchmark):
    """Benchmark loader for HumanEval Python code generation."""

    @property
    def benchmark_id(self) -> str:
        return "humaneval"

    @property
    def domain(self) -> str:
        return "coding"

    def load_examples(
        self,
        n_samples: int,
        random_seed: int = 42,
    ) -> list[HumanEvalExample]:
        """Load examples from HumanEval dataset.

        Args:
            n_samples: Number of examples to load
            random_seed: Random seed for reproducibility

        Returns:
            List of HumanEvalExample objects
        """
        # Load dataset from HuggingFace
        dataset = load_dataset("openai/openai_humaneval", split="test")

        # Sample examples
        random.seed(random_seed)
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

        examples = []
        for i, idx in enumerate(indices):
            item = dataset[idx]

            examples.append(HumanEvalExample(
                idx=i,
                task_id=item["task_id"],
                question=item["prompt"],  # Use prompt as question for consistency
                prompt=item["prompt"],
                canonical_solution=item["canonical_solution"],
                test=item["test"],
                entry_point=item["entry_point"],
            ))

        return examples

    # Formatting instruction to ensure clean output for AST parsing
    FORMAT_INSTRUCTION = (
        "Return only the completed function in a single ``` code block. "
        "Do not include test cases, assertions, or example usage within the code block."
    )

    def make_baseline_prompt(
        self,
        example: HumanEvalExample,
        instruction: str = "",
    ) -> str:
        """Create baseline prompt for a HumanEval problem.

        Args:
            example: The HumanEval example
            instruction: Optional instruction to prepend

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        if instruction:
            prompt_parts.append(instruction)

        prompt_parts.append("Complete the following Python function:")
        prompt_parts.append("")
        prompt_parts.append(example.prompt)
        prompt_parts.append("")
        prompt_parts.append(self.FORMAT_INSTRUCTION)

        return "\n".join(prompt_parts)

    def get_problem_for_attribution(self, example: HumanEvalExample) -> str:
        """Get the problem text formatted for display in the self-modeling prompt.

        Includes the benchmark-specific framing and FORMAT_INSTRUCTION.

        Args:
            example: The HumanEval example

        Returns:
            Problem text with coding-specific framing
        """
        return f"""Complete the following Python function:

{example.prompt}

{self.FORMAT_INSTRUCTION}"""

    def make_lever_prompt(
        self,
        example: HumanEvalExample,
        lever_instruction: str,
        baseline_instruction: str = "",
    ) -> str:
        """Create lever prompt for a HumanEval problem.

        The lever instruction is prepended to the prompt.

        Args:
            example: The HumanEval example
            lever_instruction: The lever instruction to add
            baseline_instruction: Ignored (lever replaces baseline)

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        if lever_instruction:
            prompt_parts.append(lever_instruction)
            prompt_parts.append("")

        prompt_parts.append("Complete the following Python function:")
        prompt_parts.append("")
        prompt_parts.append(example.prompt)
        prompt_parts.append("")
        prompt_parts.append(self.FORMAT_INSTRUCTION)

        return "\n".join(prompt_parts)
