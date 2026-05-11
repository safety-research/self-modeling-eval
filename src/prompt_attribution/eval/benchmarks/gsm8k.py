"""
Module: prompt_attribution/benchmarks/gsm8k.py

Structure:
- GSM8KExample: Dataclass for GSM8K examples
- GSM8KBenchmark: Benchmark loader for GSM8K math problems
"""

import random
from dataclasses import dataclass

from datasets import load_dataset

from .base import BaseBenchmark, Example


@dataclass
class GSM8KExample(Example):
    """GSM8K math problem example."""
    ground_truth_answer: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "idx": self.idx,
            "question": self.question,
            "ground_truth_answer": self.ground_truth_answer,
        }


class GSM8KBenchmark(BaseBenchmark):
    """Benchmark loader for GSM8K grade school math problems."""

    @property
    def benchmark_id(self) -> str:
        return "gsm8k"

    @property
    def domain(self) -> str:
        return "math"

    def load_examples(
        self,
        n_samples: int,
        random_seed: int = 42,
    ) -> list[GSM8KExample]:
        """Load examples from GSM8K dataset.

        Args:
            n_samples: Number of examples to load
            random_seed: Random seed for reproducibility

        Returns:
            List of GSM8KExample objects
        """
        # Load dataset from HuggingFace
        dataset = load_dataset("openai/gsm8k", "main", split="test")

        # Sample examples
        random.seed(random_seed)
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

        examples = []
        for i, idx in enumerate(indices):
            item = dataset[idx]
            # GSM8K answer format: "...#### <final_answer>"
            answer_text = item["answer"]
            if "####" in answer_text:
                final_answer = answer_text.split("####")[-1].strip()
            else:
                final_answer = answer_text.strip()

            examples.append(GSM8KExample(
                idx=i,
                question=item["question"],
                ground_truth_answer=final_answer,
            ))

        return examples

    def make_baseline_prompt(
        self,
        example: GSM8KExample,
        instruction: str = "",
    ) -> str:
        """Create baseline prompt for a GSM8K problem.

        Args:
            example: The GSM8K example
            instruction: Optional instruction to append

        Returns:
            Formatted prompt string
        """
        prompt = f"""Solve the problem. Put your final numerical answer in \\boxed{{}}.

Problem:
{example.question}"""

        if instruction:
            prompt += f"\n\n{instruction}"

        return prompt

    def make_lever_prompt(
        self,
        example: GSM8KExample,
        lever_instruction: str,
        baseline_instruction: str = "",
    ) -> str:
        """Create lever prompt for a GSM8K problem.

        The lever instruction replaces the baseline instruction.

        Args:
            example: The GSM8K example
            lever_instruction: The lever instruction to add
            baseline_instruction: Ignored (lever replaces baseline)

        Returns:
            Formatted prompt string
        """
        prompt = f"""Solve the problem. Put your final numerical answer in \\boxed{{}}.

Problem:
{example.question}"""

        if lever_instruction:
            prompt += f"\n\n{lever_instruction}"

        return prompt

    def get_problem_for_attribution(self, example: GSM8KExample) -> str:
        """Get the problem text formatted for display in the self-modeling prompt.

        Includes the benchmark-specific framing with boxed answer format.

        Args:
            example: The GSM8K example

        Returns:
            Problem text with math-specific framing
        """
        return f"""Solve the problem. Put your final numerical answer in \\boxed{{}}.

Problem:
{example.question}"""
