"""
Rubric compiler: CSV/document → validated JSON rules.

Reads a rubric in any format (clean CSV, composite-annotated CSV, or broken
PDF-export), sends it to an LLM for structured extraction, and validates the
output against the Pydantic schema. Invalid output raises — it does not get
silently accepted.

This is Stage 1 of the pattern described in the blog post. In production,
this ran once per rubric onboarding. Here it runs on example files.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from rubric_eval.llm import LLMClient, get_llm_client
from rubric_eval.models import CompiledRubric, Rule

# Prompt lives in a separate file so it can be reviewed and edited
# without touching code.
_PROMPT_PATH = Path(__file__).parent / "prompts" / "compile_rubric.md"


def _load_prompt() -> str:
    """Load the compilation prompt from disk."""
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Compilation prompt not found at {_PROMPT_PATH}. "
            "The prompts/ directory must be present alongside the source."
        )
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _read_raw(csv_path: str) -> str:
    """Read the rubric file as raw text.

    Not using pandas or csv module intentionally. The whole point is that
    some of these files are not valid CSVs — they are broken documents
    masquerading as tables. Raw text lets the LLM handle the mess.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Rubric file not found: {csv_path}")
    return path.read_text(encoding="utf-8")


def _parse_rules(llm_output: str) -> list[Rule]:
    """Parse LLM output into validated Rule objects.

    The LLM should return a JSON array. If it wraps the output in
    markdown fencing (```json ... ```), we strip that first.
    """
    text = llm_output.strip()

    # Strip markdown code fencing if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        raw_rules = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM output is not valid JSON. First 200 chars: {text[:200]!r}"
        ) from e

    if not isinstance(raw_rules, list):
        raise ValueError(
            f"Expected a JSON array of rules, got {type(raw_rules).__name__}"
        )

    # Validate each rule through Pydantic. This is where bad extractions
    # get caught — missing fields, wrong types, invalid enums.
    rules = []
    for i, raw in enumerate(raw_rules):
        try:
            rules.append(Rule.model_validate(raw))
        except ValidationError as e:
            raise ValidationError.from_exception_data(
                title=f"Rule {i} validation failed",
                line_errors=e.errors(),
            ) from e

    return rules


def compile_rubric(
    csv_path: str,
    client: LLMClient | None = None,
    provider: str | None = None,
) -> CompiledRubric:
    """Compile a rubric file into validated, structured rules.

    Args:
        csv_path: Path to the rubric CSV/document.
        client: Optional pre-configured LLM client. If None, one is
                created from the provider argument.
        provider: LLM provider name (anthropic/openai/vllm).

    Returns:
        A CompiledRubric with validated rules.

    Raises:
        FileNotFoundError: If the rubric file or prompt file is missing.
        ValueError: If the LLM output is not valid JSON.
        ValidationError: If any extracted rule fails schema validation.
    """
    if client is None:
        client = get_llm_client(provider)

    system_prompt = _load_prompt()
    raw_text = _read_raw(csv_path)

    llm_output = client.complete(
        system=system_prompt,
        user=f"Extract rules from this rubric document:\n\n{raw_text}",
    )

    rules = _parse_rules(llm_output)

    rubric = CompiledRubric(
        rules=rules,
        source_file=csv_path,
        compiled_at=datetime.now(),
    )

    print(f"Compiled {len(rules)} rules from {csv_path}", file=sys.stderr)
    return rubric


def main():
    """CLI entry point: compile a rubric and print JSON to stdout."""
    parser = argparse.ArgumentParser(
        description="Compile a rubric CSV into structured JSON rules."
    )
    parser.add_argument("csv_path", help="Path to the rubric CSV/document.")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "vllm"],
        default=None,
        help="LLM provider (default: anthropic, or LLM_PROVIDER env var).",
    )
    args = parser.parse_args()

    rubric = compile_rubric(args.csv_path, provider=args.provider)
    print(rubric.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
