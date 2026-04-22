"""
Brute-force evaluator: run every rule against every chunk of a document.

This is intentionally inefficient. Every rule runs on every chunk regardless
of relevance. The blog post describes this as the part to fix first in a
rebuild. This implementation uses it as the pattern to measure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tiktoken
from pydantic import ValidationError

from rubric_eval.llm import LLMClient, get_llm_client
from rubric_eval.models import (
    ChunkEvaluation,
    CompiledRubric,
    Document,
    DocumentEvaluation,
)

CHUNK_SIZE_TOKENS = 1024
_PROMPT_PATH = Path(__file__).parent / "prompts" / "evaluate_chunk.md"
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _load_prompt() -> str:
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"Evaluation prompt not found at {_PROMPT_PATH}.")
    return _PROMPT_PATH.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS) -> list[str]:
    """Split text into chunks of approximately `chunk_size` tokens.

    Uses tiktoken for token counting. No overlap — each token appears
    in exactly one chunk.
    """
    tokens = _ENCODING.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(_ENCODING.decode(chunk_tokens))
    return chunks


def _parse_chunk_evaluation(
    llm_output: str, rule_id: str, chunk_index: int
) -> ChunkEvaluation:
    """Parse a single chunk evaluation from LLM output."""
    text = llm_output.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM output for rule {rule_id}, chunk {chunk_index} is not "
            f"valid JSON. First 200 chars: {text[:200]!r}"
        ) from e

    data["chunk_index"] = chunk_index
    if data.get("rule_id") != rule_id:
        data["rule_id"] = rule_id

    return ChunkEvaluation.model_validate(data)


def evaluate_document(
    document: Document,
    rubric: CompiledRubric,
    client: LLMClient | None = None,
    provider: str | None = None,
    chunk_size: int = CHUNK_SIZE_TOKENS,
) -> DocumentEvaluation:
    """Evaluate a document against all rules in a rubric.

    Brute force: every rule * every chunk = one LLM call each.
    No filtering. No relevance routing.
    """
    if client is None:
        client = get_llm_client(provider)

    system_prompt = _load_prompt()
    full_text = "\n\n".join(
        f"## {s.heading}\n\n{s.content}" for s in document.sections
    )

    chunks = chunk_text(full_text, chunk_size)
    total_chunks = len(chunks)
    total_rules = len(rubric.rules)
    total_calls = total_chunks * total_rules

    print(
        f"Evaluating '{document.doc_id}': "
        f"{total_chunks} chunks x {total_rules} rules = {total_calls} LLM calls",
        file=sys.stderr,
    )

    evaluations: list[ChunkEvaluation] = []
    call_count = 0

    for chunk_idx, chunk in enumerate(chunks):
        for rule in rubric.rules:
            rule_json = rule.model_dump_json(indent=2)
            user_msg = (
                f"Rule:\n{rule_json}\n\n"
                f"Chunk {chunk_idx + 1} of {total_chunks}:\n\n{chunk}"
            )
            llm_output = client.complete(system=system_prompt, user=user_msg)
            try:
                ev = _parse_chunk_evaluation(llm_output, rule.rule_id, chunk_idx)
                evaluations.append(ev)
            except (ValueError, ValidationError) as e:
                print(
                    f"  Warning: parse failed for rule {rule.rule_id}, "
                    f"chunk {chunk_idx}: {e}",
                    file=sys.stderr,
                )
            call_count += 1

    print(
        f"Evaluated {call_count} chunk-rule pairs for '{document.doc_id}'",
        file=sys.stderr,
    )

    return DocumentEvaluation(
        doc_id=document.doc_id,
        evaluations=evaluations,
        total_chunks=total_chunks,
        total_rules=total_rules,
        total_llm_calls=call_count,
    )


def main():
    """CLI: evaluate a document against a compiled rubric."""
    parser = argparse.ArgumentParser(
        description="Evaluate a document against a compiled rubric."
    )
    parser.add_argument("document_path", help="Path to document JSON.")
    parser.add_argument("rubric_path", help="Path to compiled rubric JSON.")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "vllm"],
        default=None,
        help="LLM provider.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE_TOKENS,
        help=f"Chunk size in tokens (default: {CHUNK_SIZE_TOKENS}).",
    )
    args = parser.parse_args()

    doc_data = json.loads(Path(args.document_path).read_text(encoding="utf-8"))
    document = Document.model_validate(doc_data)

    rubric_data = json.loads(Path(args.rubric_path).read_text(encoding="utf-8"))
    rubric = CompiledRubric.model_validate(rubric_data)

    result = evaluate_document(
        document, rubric, provider=args.provider, chunk_size=args.chunk_size
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
