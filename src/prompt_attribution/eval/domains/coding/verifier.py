"""
Module: prompt_attribution/domains/coding/verifier.py

Structure:
- CodeVerifier: Verifier for coding problems using AST feature analysis
"""

import re
from typing import Any, Optional

from ..base import BaseVerifier
from .ast_features import ASTFeatures, extract_ast_features, extract_features_from_body


class CodeVerifier(BaseVerifier):
    """Verifier for coding problems using AST feature analysis.

    """

    def __init__(self, target_features: Optional[list[str]] = None):
        """Initialize verifier.

        Args:
            target_features: List of AST feature names to track for comparison.
                           If None, defaults to ["has_print"]
        """
        self.target_features = target_features or ["has_print"]

    def parse_answer(
        self, raw_output: str
    ) -> Optional[str]:
        """Extract answer from model output.

        For code answers, tries in order:
        1. Markdown code block (```python ... ```)
        2. Markdown code block (``` ... ```)
        3. Heuristic line-by-line extraction


        Args:
            raw_output: Full model response text

        Returns:
            Extracted answer string, or None
        """
        if not raw_output:
            return None


        # 1. Try markdown python code block (use LAST match - model may output multiple blocks)
        matches = re.findall(r'```python\s*(.*?)\s*```', raw_output, re.DOTALL)
        if matches:
            return matches[-1].strip()

        # 2. Try generic markdown code block (use LAST match)
        matches = re.findall(r'```\s*(.*?)\s*```', raw_output, re.DOTALL)
        if matches:
            return matches[-1].strip()

        # 3. Heuristic: look for lines that look like code
        lines = raw_output.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            stripped = line.strip()

            # Start of code-like content
            if (stripped.startswith('def ') or
                stripped.startswith('class ') or
                stripped.startswith('import ') or
                stripped.startswith('from ') or
                (in_code and (line.startswith('    ') or line.startswith('\t') or not stripped))):
                in_code = True
                code_lines.append(line)
            elif in_code and stripped and not line[0].isspace():
                # End of indented block
                break

        if code_lines:
            return '\n'.join(code_lines)

        # Fallback: return entire output
        return raw_output.strip()

    def extract_features(
        self,
        code: str,
        entry_point: str,
        function_prompt: Optional[str] = None,
    ) -> Optional[ASTFeatures]:
        """Extract AST features from code.

        Args:
            code: Python code string
            entry_point: Name of the function to analyze
            function_prompt: Optional function signature/docstring to combine with body

        Returns:
            ASTFeatures dataclass or None if parsing fails
        """
        if function_prompt:
            return extract_features_from_body(function_prompt, code, entry_point)
        return extract_ast_features(code, entry_point)

    def answers_match(
        self,
        answer1: Optional[str],
        answer2: Optional[str],
        entry_point: str = "",
        function_prompt: Optional[str] = None,
    ) -> bool:
        """Compare answers for equivalence.

        For code answers, compares by AST features.

        Args:
            answer1: First answer (code string or MCQ letter)
            answer2: Second answer (code string or MCQ letter)
            entry_point: Function name to analyze (for code)
            function_prompt: Optional function signature (for code)

        Returns:
            True if answers are equivalent
        """
        if answer1 is None or answer2 is None:
            return answer1 == answer2


        features1 = self.extract_features(answer1, entry_point, function_prompt)
        features2 = self.extract_features(answer2, entry_point, function_prompt)

        if features1 is None or features2 is None:
            # If we can't parse features, assume no flip (can't determine)
            return True

        # Check only target features
        for feature in self.target_features:
            if getattr(features1, feature, None) != getattr(features2, feature, None):
                return False
        return True

    def check_feature(
        self,
        answer: str,
        parsed_answer: Any = None,
        perturbation: Any = None,
        example: Any = None,
        **kwargs: Any,
    ) -> bool:
        """Check if a specific feature is present in the code.

        Used by self-modeling eval 9 (feature presence probability).
        Checks the first target feature defined by the perturbation config.

        Args:
            answer: Raw response text
            parsed_answer: Parsed code string (from parse_answer)
            perturbation: PerturbationConfig with target_features
            example: HumanEvalExample with entry_point and prompt

        Returns:
            True if the target feature is present
        """
        if not parsed_answer or not perturbation:
            return False

        target = perturbation.target_features[0] if perturbation.target_features else None
        if not target:
            return False

        entry_point = getattr(example, "entry_point", "") if example else ""
        function_prompt = getattr(example, "prompt", None) if example else None

        features = self.extract_features(str(parsed_answer), entry_point, function_prompt)
        if features is None:
            return False

        return bool(getattr(features, target, False))

    # Natural language descriptions for features
    FEATURE_PHRASES = {
        'has_print': 'print statements',
        'has_try_except': 'try/except blocks',
        'has_loop': 'loops',
        'has_comprehension': 'comprehensions/generators',
        'has_assert': 'assert statements',
        'has_nested_function': 'nested functions',
        'uses_recursion': 'recursion',
        'uses_builtin': 'built-in functions',
        'uses_lambda': 'lambda expressions',
        'uses_global': 'global variables',
        'uses_type_hints': 'type hints',
        'mutates_input': 'input mutation',
        'raises_exception': 'raise statements',
    }

