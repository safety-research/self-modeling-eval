"""
Module: prompt_attribution/benchmarks/wildguardtest.py

Structure:
- WildGuardTestExample: Dataclass for WildGuardTest examples
- WildGuardTestBenchmark: Benchmark loader for WildGuardTest prompts (mix of harmful/benign)
"""

import os
import random
from dataclasses import dataclass
from typing import Optional, Literal

from datasets import load_dataset

from .base import BaseBenchmark, Example


@dataclass
class WildGuardTestExample(Example):
    """WildGuardTest prompt example."""

    prompt_harm_label: Optional[str] = None  # "harmful", "unharmful", or None
    adversarial: bool = False  # Whether prompt uses adversarial techniques
    subcategory: str = ""  # Fine-grained risk category

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "idx": self.idx,
            "question": self.question,
            "prompt_harm_label": self.prompt_harm_label,
            "adversarial": self.adversarial,
            "subcategory": self.subcategory,
        }


class WildGuardTestBenchmark(BaseBenchmark):
    """Benchmark loader for WildGuardTest prompts.

    WildGuardTest is part of WildGuardMix from Allen AI, containing a mix of
    harmful and benign prompts for evaluating safety classifiers.

    - 945 unharmful prompts
    - 754 harmful prompts
    - 55% vanilla, 45% adversarial

    Dataset: https://huggingface.co/datasets/allenai/wildguardmix
    Paper: https://arxiv.org/abs/2406.18495

    NOTE: This is a gated dataset requiring HF_TOKEN authentication.
    """

    def __init__(self, filter_by: Optional[Literal["harmful", "unharmful"]] = None):
        """Initialize benchmark with optional filtering.

        Args:
            filter_by: If set, only load prompts with this harm label.
                       None = load all prompts (default)
                       "harmful" = only harmful prompts
                       "unharmful" = only benign prompts
        """
        super().__init__()
        self.filter_by = filter_by

    @property
    def benchmark_id(self) -> str:
        if self.filter_by:
            return f"wildguardtest_{self.filter_by}"
        return "wildguardtest"

    @property
    def domain(self) -> str:
        return "safety"

    def load_examples(
        self,
        n_samples: int,
        random_seed: int = 42,
    ) -> list[WildGuardTestExample]:
        """Load examples from WildGuardTest dataset.

        Args:
            n_samples: Number of examples to load
            random_seed: Random seed for reproducibility

        Returns:
            List of WildGuardTestExample objects
        """
        # Get HF token from environment
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "HF_TOKEN environment variable required for WildGuardMix dataset. "
                "Please set it in .env or export it."
            )

        # Load dataset from HuggingFace
        dataset = load_dataset(
            "allenai/wildguardmix",
            "wildguardtest",
            split="test",
            token=token,
        )

        # Filter by harm label if specified
        if self.filter_by:
            dataset = dataset.filter(
                lambda x: x["prompt_harm_label"] == self.filter_by
            )

        # Stratified sampling across 4 quadrants: (harm_label × adversarial)
        # Each quadrant has different flip dynamics:
        #   harmful+vanilla:     model refuses, hard to flip on strong models
        #   harmful+adversarial: model may comply (fooled), flip toward refuse
        #   unharmful+vanilla:   model complies, flip toward over-refusal
        #   unharmful+adversarial: model may over-refuse, flip either way
        random.seed(random_seed)

        if self.filter_by is None and n_samples >= 4:
            buckets: dict[str, list[int]] = {
                "harmful_vanilla": [],
                "harmful_adversarial": [],
                "unharmful_vanilla": [],
                "unharmful_adversarial": [],
            }
            for i in range(len(dataset)):
                harm = dataset[i]["prompt_harm_label"]
                adv = dataset[i].get("adversarial", False)
                key = f"{harm}_{'adversarial' if adv else 'vanilla'}"
                if key in buckets:
                    buckets[key].append(i)

            for v in buckets.values():
                random.shuffle(v)

            # Distribute evenly across quadrants, remainder goes to largest buckets
            per_bucket = n_samples // 4
            remainder = n_samples % 4

            indices = []
            for key in ["harmful_vanilla", "harmful_adversarial",
                        "unharmful_vanilla", "unharmful_adversarial"]:
                n = per_bucket + (1 if remainder > 0 else 0)
                remainder = max(0, remainder - 1)
                indices.extend(buckets[key][:min(n, len(buckets[key]))])

            random.shuffle(indices)
        else:
            indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

        examples = []
        for i, idx in enumerate(indices):
            item = dataset[idx]

            examples.append(
                WildGuardTestExample(
                    idx=i,
                    question=item["prompt"],
                    prompt_harm_label=item.get("prompt_harm_label"),
                    adversarial=item.get("adversarial", False),
                    subcategory=item.get("subcategory", ""),
                )
            )

        return examples

    def make_baseline_prompt(
        self,
        example: WildGuardTestExample,
        instruction: str = "",
    ) -> str:
        """Create baseline prompt.

        For safety domain, instruction is placed BEFORE the prompt (prefix framing).

        Args:
            example: The WildGuardTest example
            instruction: Optional instruction to prepend

        Returns:
            Formatted prompt string
        """
        if instruction:
            return f"{instruction}\n\n{example.question}"
        return example.question

    def make_lever_prompt(
        self,
        example: WildGuardTestExample,
        lever_instruction: str,
        baseline_instruction: str = "",
    ) -> str:
        """Create lever prompt.

        For safety domain, lever is placed BEFORE the prompt (prefix framing).

        Args:
            example: The WildGuardTest example
            lever_instruction: The lever instruction to add (placed before prompt)
            baseline_instruction: Ignored (lever replaces baseline)

        Returns:
            Formatted prompt string
        """
        if lever_instruction:
            return f"{lever_instruction}\n\n{example.question}"
        return example.question

    def get_problem_for_attribution(self, example: WildGuardTestExample) -> str:
        """Get the problem text formatted for display in the self-modeling prompt.

        Args:
            example: The WildGuardTest example

        Returns:
            The prompt text
        """
        return example.question
