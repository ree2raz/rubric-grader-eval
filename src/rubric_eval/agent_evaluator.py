"""
Agentic evaluator: a single agent with targeted tools evaluates a document
against a rubric.

This implements the architecture described in the blog post: read metadata,
list sections, fetch targeted sections, look up rules, submit evaluations.
The agent decides what to read instead of brute-forcing every rule against
every chunk.

Post-hoc, batch, per-document. NOT real-time. Same temporal posture as the
rest of the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from rubric_eval.llm import LLMClient, get_llm_client
from rubric_eval.models import (
    ChunkEvaluation,
    CompiledRubric,
    Document,
    DocumentEvaluation,
    Rule,
    Section,
)

MAX_TOOL_CALLS = 50
_PROMPT_PATH = Path(__file__).parent / "prompts" / "agent_evaluator.md"


class AgentAction(BaseModel):
    """One tool call the agent wants to make."""

    type: str = Field(
        description="One of: get_document_metadata, list_sections, fetch_section, get_rule, submit_evaluations, finish"
    )
    params: dict = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """The agent's output each turn: reasoning + action."""

    thought: str
    action: AgentAction


class AgentState:
    """Mutable state the agent accumulates across turns."""

    def __init__(self, document: Document, rubric: CompiledRubric, max_calls: int = MAX_TOOL_CALLS):
        self.document = document
        self.rubric = rubric
        self.max_calls = max_calls
        self.call_count = 0
        self.submitted: dict[str, ChunkEvaluation] = {}
        self.read_sections: set[int] = set()
        self.finished = False
        self.history: list[dict] = []
        self.known_sections: list[dict] | None = None
        self.known_rules: list[dict] | None = None

    def budget_remaining(self) -> int:
        return self.max_calls - self.call_count

    def all_rules_submitted(self) -> bool:
        rubric_ids = {r.rule_id for r in self.rubric.rules}
        return rubric_ids.issubset(self.submitted.keys())


def _load_prompt() -> str:
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"Agent prompt not found at {_PROMPT_PATH}.")
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _build_state_message(state: AgentState) -> str:
    """Build a user message describing the current state for the LLM."""
    lines: list[str] = []
    lines.append(f"Document ID: {state.document.doc_id}")
    lines.append(f"Tool calls used: {state.call_count} / {state.max_calls}")
    lines.append(f"Rules submitted: {len(state.submitted)} / {len(state.rubric.rules)}")

    if state.read_sections:
        lines.append(f"Sections read: {sorted(state.read_sections)}")
    else:
        lines.append("Sections read: none yet")

    if state.submitted:
        lines.append("Submitted evaluations:")
        for rid, ev in sorted(state.submitted.items()):
            lines.append(f"  {rid}: {ev.verdict}")

    remaining = [r.rule_id for r in state.rubric.rules if r.rule_id not in state.submitted]
    if remaining:
        lines.append(f"Remaining rules: {remaining}")
    else:
        lines.append("All rules submitted.")

    if state.known_sections:
        lines.append("\n[MEMORY] Known Document Sections:")
        lines.append(json.dumps(state.known_sections, indent=2))

    if state.known_rules:
        lines.append("\n[MEMORY] Known Rubric Rules:")
        lines.append(json.dumps(state.known_rules, indent=2))

    if state.history:
        lines.append("\n--- CONVERSATION HISTORY ---")
        for turn in state.history[-5:]: # Keep the last 5 turns to avoid blowing context
            lines.append(f"Action: {turn['action']}")
            lines.append(f"Result: {json.dumps(turn['result'], indent=2)}")
        lines.append("-----------------------------")

    return "\n".join(lines)


def _execute_action(state: AgentState, action: AgentAction) -> dict:
    """Execute one agent action and return the result as a dict."""
    action_type = action.type
    params = action.params

    if action_type == "get_document_metadata":
        meta = state.document.metadata
        return {
            "doc_type": meta.doc_type,
            "project": meta.project,
            "version": meta.version,
            "author_hash": meta.author_hash,
        }

    if action_type == "list_sections":
        secs = [{"index": s.index, "heading": s.heading} for s in state.document.sections]
        state.known_sections = secs
        return {"sections": secs}

    if action_type == "fetch_section":
        idx = params.get("index")
        if idx is None:
            return {"error": "Missing 'index' param"}
        try:
            idx = int(idx)
        except ValueError:
            return {"error": f"Invalid 'index' param: {idx}. Must be an integer."}
        for s in state.document.sections:
            if s.index == idx:
                state.read_sections.add(idx)
                # Truncate content if it's absurdly long to prevent context blowouts,
                # though realistically chunks should be manageable.
                content = s.content
                if len(content) > 4000:
                    content = content[:4000] + "\n...[TRUNCATED]"
                return {"index": s.index, "heading": s.heading, "content": content}
        return {"error": f"Section index {idx} not found"}

    if action_type == "get_rule":
        rule_id = params.get("rule_id")
        if rule_id is None:
            # Return all rules if no specific rule_id requested
            rules = [r.model_dump() for r in state.rubric.rules]
            state.known_rules = rules
            return {"rules": rules}
        for r in state.rubric.rules:
            if r.rule_id == rule_id:
                return r.model_dump()
        return {"error": f"Rule {rule_id} not found"}

    if action_type == "submit_evaluations":
        evals = params.get("evaluations", [])
        if not isinstance(evals, list):
            return {"error": "Missing or invalid 'evaluations' param. Must be a list."}
        
        results = []
        for ev_data in evals:
            rule_id = ev_data.get("rule_id")
            verdict = ev_data.get("verdict")
            if isinstance(verdict, str):
                verdict = verdict.lower()
            evidence = ev_data.get("evidence", "")
            
            if rule_id is None or verdict is None:
                results.append({"rule_id": rule_id, "error": "Missing rule_id or verdict"})
                continue
            if verdict not in ("pass", "fail"):
                results.append({"rule_id": rule_id, "error": f"Invalid verdict: {verdict}"})
                continue

            ev = ChunkEvaluation(
                rule_id=rule_id,
                chunk_index=0,
                verdict=verdict,
                reasoning=str(state.call_count),  # placeholder: call count as string
                evidence_quote=evidence,
            )
            state.submitted[rule_id] = ev
            results.append({"rule_id": rule_id, "status": "submitted", "verdict": verdict})
            
        return {"status": "batch_submitted", "results": results}

    if action_type == "finish":
        state.finished = True
        return {"status": "finished"}

    return {"error": f"Unknown action type: {action_type}"}


def _parse_agent_response(llm_output: str) -> AgentResponse:
    """Parse the LLM's JSON output into an AgentResponse."""
    text = llm_output.strip()

    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Agent output is not valid JSON. First 300 chars: {text[:300]!r}") from e

    return AgentResponse.model_validate(data)


def evaluate_document_agentic(
    document: Document,
    rubric: CompiledRubric,
    client: LLMClient | None = None,
    provider: str | None = None,
    max_calls: int = MAX_TOOL_CALLS,
) -> DocumentEvaluation:
    """Evaluate a document using an agent with targeted tools.

    The agent decides which sections to read and which rules to evaluate,
    rather than brute-forcing every rule against every chunk.
    """
    if client is None:
        client = get_llm_client(provider)

    system_prompt = _load_prompt()
    state = AgentState(document, rubric, max_calls=max_calls)

    print(
        f"Agent evaluating '{document.doc_id}': {len(rubric.rules)} rules, "
        f"budget {max_calls} calls",
        file=sys.stderr,
    )

    # Agent loop
    while state.call_count < state.max_calls and not state.finished:
        state_msg = _build_state_message(state)

        try:
            llm_output = client.complete(system=system_prompt, user=state_msg)
        except Exception as e:
            print(f"  LLM call failed at step {state.call_count}: {e}", file=sys.stderr)
            break

        state.call_count += 1

        try:
            response = _parse_agent_response(llm_output)
            print(f"\n[Step {state.call_count}] THOUGHT: {response.thought}", file=sys.stderr)
            print(f"[Step {state.call_count}] ACTION: {response.action.type}({response.action.params})", file=sys.stderr)
        except (ValueError, ValidationError) as e:
            print(f"  Warning: parse failed at step {state.call_count}: {e}", file=sys.stderr)
            state.last_action_result = {
                "error": f"Failed to parse your response: {e}. Please ensure you output valid JSON matching the schema."
            }
            continue

        # Execute the action
        result = _execute_action(state, response.action)
        state.history.append({
            "action": f"{response.action.type}({response.action.params})",
            "result": result
        })

        # If agent tried to finish or submitted all rules, check
        if state.finished or state.all_rules_submitted():
            break

    # Build DocumentEvaluation from submitted evaluations
    # For rules not submitted, default to "fail"
    evaluations: list[ChunkEvaluation] = []
    for rule in rubric.rules:
        if rule.rule_id in state.submitted:
            evaluations.append(state.submitted[rule.rule_id])
        else:
            evaluations.append(
                ChunkEvaluation(
                    rule_id=rule.rule_id,
                    chunk_index=0,
                    verdict="fail",
                    reasoning="Agent did not submit an evaluation within budget",
                    evidence_quote="",
                )
            )

    print(
        f"Agent finished '{document.doc_id}': {state.call_count} calls, "
        f"{len(state.submitted)} rules evaluated",
        file=sys.stderr,
    )

    return DocumentEvaluation(
        doc_id=document.doc_id,
        evaluations=evaluations,
        total_chunks=len(document.sections),
        total_rules=len(rubric.rules),
        total_llm_calls=state.call_count,
    )


def main():
    """CLI: agentic evaluation of a document against a compiled rubric."""
    parser = argparse.ArgumentParser(
        description="Agentic evaluation: targeted tool use instead of brute force."
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
        "--max-calls",
        type=int,
        default=MAX_TOOL_CALLS,
        help=f"Max tool calls budget (default: {MAX_TOOL_CALLS}).",
    )
    args = parser.parse_args()

    doc_data = json.loads(Path(args.document_path).read_text(encoding="utf-8"))
    document = Document.model_validate(doc_data)

    rubric_data = json.loads(Path(args.rubric_path).read_text(encoding="utf-8"))
    rubric = CompiledRubric.model_validate(rubric_data)

    result = evaluate_document_agentic(
        document, rubric, provider=args.provider, max_calls=args.max_calls
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
