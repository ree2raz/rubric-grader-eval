"""
rubric-grader-eval: A reference pattern for compiling unstructured rubrics into
machine-readable schemas, then evaluating documents against them.

The compiler is the artifact — handles clean CSV, boolean composites, and
document masquerades. The evaluator and eval harness measure whether the
compilation produced rules that can be evaluated correctly.
"""

__version__ = "0.1.0"
