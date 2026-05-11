"""
Module: prompt_attribution/benchmarks/bbq.py

Structure:
- BBQExample: Dataclass for BBQ bias benchmark examples
- BBQBenchmark: Benchmark loader for BBQ (Bias Benchmark for QA)
"""

import random
from dataclasses import dataclass
from typing import Optional

from datasets import concatenate_datasets, load_dataset

from .base import BaseBenchmark, Example


@dataclass
class BBQExample(Example):
    """BBQ bias benchmark example.

    BBQ tests social biases through ambiguous and disambiguated contexts
    with multiple-choice answers.
    """

    idx: int = 0
    question: str = ""
    context: str = ""
    ans0: str = ""
    ans1: str = ""
    ans2: str = ""
    label: int = 0  # Correct answer index (0, 1, or 2)
    category: str = ""  # e.g., "Age", "Gender_identity", "Race_ethnicity"
    context_condition: str = ""  # "ambig" or "disambig"
    question_polarity: str = ""  # "neg" or "nonneg"
    example_id: int = 0  # Original BBQ example ID

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "idx": self.idx,
            "question": self.question,
            "context": self.context,
            "ans0": self.ans0,
            "ans1": self.ans1,
            "ans2": self.ans2,
            "label": self.label,
            "category": self.category,
            "context_condition": self.context_condition,
            "question_polarity": self.question_polarity,
            "example_id": self.example_id,
        }


class BBQBenchmark(BaseBenchmark):
    """Benchmark loader for BBQ (Bias Benchmark for QA).

    BBQ tests social biases through multiple-choice questions with
    ambiguous and disambiguated contexts. In ambiguous contexts, the
    correct answer is typically "Can't be determined", but biased models
    may select stereotyped answers.

    Dataset: heegyu/bbq on HuggingFace
    """

    @property
    def benchmark_id(self) -> str:
        return "bbq"

    @property
    def domain(self) -> str:
        return "fairness"

    def __init__(
        self,
        category: Optional[str] = None,
        context_condition: Optional[str] = None,
    ):
        """Initialize BBQ benchmark with optional filters.

        Args:
            category: Filter to specific bias category (e.g., "Age", "Gender_identity")
            context_condition: Filter to "ambig" or "disambig" contexts
        """
        self.category = category
        self.context_condition = context_condition

    def load_examples(
        self,
        n_samples: int,
        random_seed: int = 42,
    ) -> list[BBQExample]:
        """Load examples from BBQ dataset (lighteval/bbq_helm).

        Args:
            n_samples: Number of examples to load
            random_seed: Random seed for reproducibility

        Returns:
            List of BBQExample objects
        """
        # Load all categories from lighteval/bbq_helm
        all_configs = [
            "Age", "Disability_status", "Gender_identity", "Nationality",
            "Physical_appearance", "Race_ethnicity", "Race_x_gender",
            "Race_x_SES", "Religion", "SES", "Sexual_orientation",
        ]
        datasets_list = []
        config_labels = []
        for cfg_name in all_configs:
            try:
                ds = load_dataset("lighteval/bbq_helm", cfg_name, split="test")
                datasets_list.append(ds)
                config_labels.extend([cfg_name] * len(ds))
            except Exception:
                continue
        dataset = concatenate_datasets(datasets_list)

        # Apply filters
        filtered_indices = []
        for i in range(len(dataset)):
            cat = config_labels[i]
            if self.category and cat.lower().replace("_", "") != self.category.lower().replace("_", ""):
                continue
            # Extract context_condition from tags
            item = dataset[i]
            cc = self._extract_context_condition(item)
            if self.context_condition and cc != self.context_condition:
                continue
            filtered_indices.append(i)

        # Sample examples
        random.seed(random_seed)
        sample_size = min(n_samples, len(filtered_indices))
        sampled_indices = random.sample(filtered_indices, sample_size)

        examples = []
        for i, idx in enumerate(sampled_indices):
            item = dataset[idx]
            choices = item["choices"]
            cat = config_labels[idx]
            cc = self._extract_context_condition(item)
            qp = self._extract_question_polarity(item)

            # Build the question string (context + question)
            full_question = f"{item['context']}\n\n{item['question']}"

            examples.append(
                BBQExample(
                    idx=i,
                    question=full_question,
                    context=item["context"],
                    ans0=choices[0],
                    ans1=choices[1],
                    ans2=choices[2],
                    label=item["gold_index"],
                    category=cat,
                    context_condition=cc,
                    question_polarity=qp,
                    example_id=idx,
                )
            )

        return examples

    @staticmethod
    def _extract_context_condition(item: dict) -> str:
        """Extract context_condition (ambig/disambig) from lighteval/bbq_helm tags."""
        refs = item.get("references", {})
        tags_list = refs.get("tags", [])
        for tags in tags_list:
            if not tags:
                continue
            for tag in tags:
                if tag in ("ambiguous", "disambiguous"):
                    return "ambig" if tag == "ambiguous" else "disambig"
        return ""

    @staticmethod
    def _extract_question_polarity(item: dict) -> str:
        """Extract question_polarity (neg/nonneg) from lighteval/bbq_helm tags."""
        refs = item.get("references", {})
        tags_list = refs.get("tags", [])
        for tags in tags_list:
            if not tags:
                continue
            for tag in tags:
                if tag in ("negative", "non-negative"):
                    return "neg" if tag == "negative" else "nonneg"
        return ""

    # Format instruction - allows reasoning with \boxed{} for final answer
    FORMAT_INSTRUCTION = "Put your final answer (A, B, or C) in \\boxed{}."

    def _format_options(self, example: BBQExample) -> str:
        """Format answer options as A/B/C list."""
        return f"A) {example.ans0}\nB) {example.ans1}\nC) {example.ans2}"

    def make_baseline_prompt(
        self,
        example: BBQExample,
        instruction: str = "",
    ) -> str:
        """Create baseline prompt for a BBQ question.

        Args:
            example: The BBQ example
            instruction: Optional instruction to append

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "Read the following context and answer the question by selecting A, B, or C.",
            "",
            f"Context: {example.context}",
            "",
            f"Question: {example.question.split(chr(10))[-1]}",  # Just the question part
            "",
            self._format_options(example),
        ]

        if instruction:
            prompt_parts.append("")
            prompt_parts.append(instruction)

        prompt_parts.append("")
        prompt_parts.append(self.FORMAT_INSTRUCTION)

        return "\n".join(prompt_parts)

    def get_problem_for_attribution(self, example: BBQExample) -> str:
        """Get the problem text formatted for display in the self-modeling prompt.

        Includes the full task instruction structure to match the actual prompt.

        Args:
            example: The BBQ example

        Returns:
            Problem text with task instruction, context, question, and options
        """
        return f"""Read the following context and answer the question by selecting A, B, or C.

Context: {example.context}

Question: {example.question.split(chr(10))[-1]}

{self._format_options(example)}

{self.FORMAT_INSTRUCTION}"""

    def make_lever_prompt(
        self,
        example: BBQExample,
        lever_instruction: str,
        baseline_instruction: str = "",
    ) -> str:
        """Create lever prompt for a BBQ question.

        The lever instruction is placed after the task instruction.

        Args:
            example: The BBQ example
            lever_instruction: The lever instruction to add
            baseline_instruction: Optional baseline instruction (combined with lever)

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "Read the following context and answer the question by selecting A, B, or C.",
        ]

        # Add lever instruction after task instruction (no "Important:" prefix)
        if lever_instruction:
            prompt_parts.append("")
            prompt_parts.append(lever_instruction)

        prompt_parts.extend(
            [
                "",
                f"Context: {example.context}",
                "",
                f"Question: {example.question.split(chr(10))[-1]}",
                "",
                self._format_options(example),
                "",
                self.FORMAT_INSTRUCTION,
            ]
        )

        return "\n".join(prompt_parts)
