"""
AST Feature Extraction for Code Attribution

Extracts multi-label features from Python code using AST analysis.
"""

import ast
from dataclasses import dataclass, asdict, fields
from typing import Optional


@dataclass
class ASTFeatures:
    """Multi-label features extracted from Python code."""
    has_print: bool = False           # Uses print()
    mutates_input: bool = False       # Modifies input parameters in-place
    uses_recursion: bool = False      # Function calls itself
    uses_builtin: bool = False        # Uses built-in functions
    raises_exception: bool = False    # Uses raise statement
    has_assert: bool = False          # Uses assert statement
    has_loop: bool = False            # Uses for/while loops
    has_comprehension: bool = False  # Uses list/dict/set comprehension or generator
    uses_lambda: bool = False         # Uses lambda expressions
    has_try_except: bool = False      # Uses try/except blocks
    uses_global: bool = False         # Uses global variables
    has_nested_function: bool = False # Defines nested functions
    uses_type_hints: bool = False     # Has type annotations

    def to_dict(self) -> dict[str, bool]:
        """Convert to dictionary."""
        return asdict(self)

    def diff(self, other: 'ASTFeatures') -> dict[str, tuple[bool, bool]]:
        """Return features that differ between self and other.

        Returns dict mapping feature_name -> (self_value, other_value)
        """
        if other is None:
            return {}
        diffs = {}
        for field in fields(self):
            v1 = getattr(self, field.name)
            v2 = getattr(other, field.name)
            if v1 != v2:
                diffs[field.name] = (v1, v2)
        return diffs

    def __str__(self) -> str:
        """String representation showing True features."""
        true_features = [f.name for f in fields(self) if getattr(self, f.name)]
        return f"ASTFeatures({', '.join(true_features) or 'none'})"


# Built-in functions to detect
BUILTINS = {
    'len', 'sum', 'max', 'min', 'sorted', 'reversed', 'enumerate',
    'zip', 'map', 'filter', 'range', 'abs', 'all', 'any', 'round',
    'int', 'float', 'str', 'list', 'dict', 'set', 'tuple', 'bool',
    'ord', 'chr', 'hex', 'bin', 'oct', 'pow', 'divmod',
}

# Methods that mutate input
MUTATING_METHODS = {
    'append', 'extend', 'insert', 'remove', 'pop', 'clear',
    'sort', 'reverse', 'update', 'add', 'discard',
}


class FeatureVisitor(ast.NodeVisitor):
    """AST visitor that extracts code features."""

    def __init__(self, func_name: str):
        self.func_name = func_name
        self.in_target_func = False
        self.func_params = set()
        self.features = {
            'has_print': False,
            'mutates_input': False,
            'uses_recursion': False,
            'uses_builtin': False,
            'raises_exception': False,
            'has_assert': False,
            'has_loop': False,
            'has_comprehension': False,
            'uses_lambda': False,
            'has_try_except': False,
            'uses_global': False,
            'has_nested_function': False,
            'uses_type_hints': False,
        }

    def visit_FunctionDef(self, node):
        if node.name == self.func_name:
            self.in_target_func = True
            self.func_params = {arg.arg for arg in node.args.args}

            # Check type hints
            if any(arg.annotation for arg in node.args.args) or node.returns:
                self.features['uses_type_hints'] = True

            self.generic_visit(node)
            self.in_target_func = False
        elif self.in_target_func:
            # Nested function definition
            self.features['has_nested_function'] = True
            self.generic_visit(node)
        else:
            self.generic_visit(node)

    # Also handle async function definitions
    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Call(self, node):
        if self.in_target_func:
            # Check for print
            if isinstance(node.func, ast.Name) and node.func.id == 'print':
                self.features['has_print'] = True

            # Check for builtins
            if isinstance(node.func, ast.Name) and node.func.id in BUILTINS:
                self.features['uses_builtin'] = True

            # Check for recursion
            if isinstance(node.func, ast.Name) and node.func.id == self.func_name:
                self.features['uses_recursion'] = True

            # Check for input mutation (list/dict methods)
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in MUTATING_METHODS:
                    # Check if called on a function parameter
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id in self.func_params:
                            self.features['mutates_input'] = True

        self.generic_visit(node)

    def visit_Raise(self, node):
        if self.in_target_func:
            self.features['raises_exception'] = True
        self.generic_visit(node)

    def visit_Assert(self, node):
        if self.in_target_func:
            self.features['has_assert'] = True
        self.generic_visit(node)

    def visit_For(self, node):
        if self.in_target_func:
            self.features['has_loop'] = True
        self.generic_visit(node)

    def visit_While(self, node):
        if self.in_target_func:
            self.features['has_loop'] = True
        self.generic_visit(node)

    def visit_ListComp(self, node):
        if self.in_target_func:
            self.features['has_comprehension'] = True
        self.generic_visit(node)

    def visit_DictComp(self, node):
        if self.in_target_func:
            self.features['has_comprehension'] = True
        self.generic_visit(node)

    def visit_SetComp(self, node):
        if self.in_target_func:
            self.features['has_comprehension'] = True
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        if self.in_target_func:
            self.features['has_comprehension'] = True
        self.generic_visit(node)

    def visit_Lambda(self, node):
        if self.in_target_func:
            self.features['uses_lambda'] = True
        self.generic_visit(node)

    def visit_Try(self, node):
        if self.in_target_func:
            self.features['has_try_except'] = True
        self.generic_visit(node)

    def visit_Global(self, node):
        if self.in_target_func:
            self.features['uses_global'] = True
        self.generic_visit(node)


def extract_ast_features(code: str, entry_point: str) -> Optional[ASTFeatures]:
    """
    Extract AST features from Python code.

    Args:
        code: Python source code (full code including function definition)
        entry_point: Name of the function to analyze

    Returns:
        ASTFeatures dataclass or None if parsing fails
    """
    if not code or not code.strip():
        return None

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    visitor = FeatureVisitor(entry_point)
    visitor.visit(tree)

    return ASTFeatures(**visitor.features)


def extract_features_from_body(
    function_prompt: str,
    function_body: str,
    entry_point: str,
) -> Optional[ASTFeatures]:
    """
    Extract AST features from function body.

    Combines the function signature (prompt) with the body and extracts features.

    Args:
        function_prompt: Function signature and docstring
        function_body: Function body code (will be indented)
        entry_point: Name of the function

    Returns:
        ASTFeatures dataclass or None if parsing fails
    """
    if not function_body:
        return None

    # Strip function definition if model included it (we'll use the prompt's signature)
    lines = function_body.split('\n')
    import_lines = []  # Collect imports separately
    cleaned_lines = []
    skip_docstring = False
    in_docstring = False
    found_function_def = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Collect import statements (before function definition)
        if not found_function_def and (
            stripped.startswith('import ') or
            stripped.startswith('from ')
        ):
            import_lines.append(stripped)
            continue

        # Skip empty lines before function definition
        if not found_function_def and not stripped:
            continue

        # Skip function definition line
        if stripped.startswith(f'def {entry_point}('):
            found_function_def = True
            skip_docstring = True
            continue

        # Skip docstrings that follow def (already in function_prompt)
        if skip_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring:
                    # End of docstring
                    in_docstring = False
                    skip_docstring = False
                    continue
                elif stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                    # Single-line docstring
                    continue
                else:
                    # Start of multi-line docstring
                    in_docstring = True
                    continue
            elif in_docstring:
                continue
            else:
                skip_docstring = False

        # After function def, stop at module-level (unindented) code
        # This handles cases where model adds asserts/tests outside the function
        if found_function_def and stripped and not line[0].isspace():
            # Non-empty, unindented line after function = module-level code
            break

        cleaned_lines.append(line)

    function_body = '\n'.join(cleaned_lines)

    # Ensure proper indentation for function body
    # First, normalize indentation by finding minimum indent and removing it
    lines = function_body.split('\n')

    # Find minimum indentation among non-empty lines
    min_indent = float('inf')
    for line in lines:
        if line.strip():
            leading_spaces = len(line) - len(line.lstrip())
            min_indent = min(min_indent, leading_spaces)

    if min_indent == float('inf'):
        min_indent = 0

    # Normalize: remove min_indent from all lines, then add 4 spaces
    indented_lines = []
    for line in lines:
        if line.strip():
            # Remove the minimum indentation first (handles both spaces and tabs)
            if min_indent > 0:
                # Count leading whitespace and remove min_indent characters
                leading_ws = len(line) - len(line.lstrip())
                if leading_ws >= min_indent:
                    line = line[min_indent:]
            # Then add 4 spaces for function body
            line = '    ' + line
        indented_lines.append(line)

    indented_body = '\n'.join(indented_lines)

    # Combine imports, prompt and body
    # Imports go at module level (before function definition)
    if import_lines:
        imports = '\n'.join(import_lines)
        full_code = f"{imports}\n\n{function_prompt}\n{indented_body}"
    else:
        full_code = f"{function_prompt}\n{indented_body}"

    return extract_ast_features(full_code, entry_point)
