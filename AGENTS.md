# Agent Instructions for llm-rubric-eval

## What This Repo Is

A reference pattern for compiling unstructured rubrics into machine-readable schemas, then evaluating documents against them with golden-set ground truth. Intentionally minimal. Not a product. The compiler is the artifact — it handles three variance cases that represent real-world rubric format problems. The eval harness proves the compiler worked. The repo accompanies a blog post about rubric grading in production.

## Architecture

Two-stage pipeline plus eval harness:

```
CSV rubric → [compiler.py] → compiled JSON rules → [evaluator.py] → chunk×rule verdicts → [eval.py] → metrics
```

- **compiler.py**: Reads rubric as raw text (not pandas), sends to LLM, validates output via Pydantic. Handles three variance cases: clean CSV, boolean composites in notes columns, broken PDF-exported documents. The most carefully tuned component — this is the artifact.
- **evaluator.py**: Brute-force. Chunks document by 1024 tokens (tiktoken). For each chunk × each rule, one LLM call. No filtering, no relevance routing. This is intentionally inefficient — it's the pattern to measure.
- **eval.py**: Loads documents with `ground_truth` labels, runs evaluator, compares system verdicts to ground truth. Prints per-category precision/recall/F1 and per-rule agreement table. Supports `--pre-evaluated` flag to load pre-computed verdicts from disk — no LLM calls, no API key needed.
- **llm.py**: Provider factory. Supports Anthropic (default), OpenAI, vLLM. Unified `complete(system, user) -> str` interface.
- **models.py**: Pydantic v2 models. All LLM output is validated here. Invalid output raises, never silently accepted.

## Directory Layout

```
src/rubric_eval/          # Package source
  prompts/                # .md prompt files loaded at runtime — NOT inline in code
    compile_rubric.md     # Rubric extraction prompt
    evaluate_chunk.md     # Per-chunk evaluation prompt
schema/                   # JSON schema for compiled rules
examples/
  rubrics/                # 3 CSV rubrics (clean, boolean_composite, document_masquerade)
  compiled/               # Pre-compiled rubric JSONs (from real LLM runs)
  documents/              # 5 synthetic tech docs with ground_truth labels
tests/                    # pytest tests — all mocked, no API calls
```

## Key Conventions

### Code Style
- Python 3.11+. Type hints everywhere. `from __future__ import annotations`.
- Pydantic v2 for all data models. Use `model_validate()`, not `parse_obj()`.
- No pandas. CSV files are read as raw text intentionally — some are broken documents.
- Prompts live in `src/rubric_eval/prompts/*.md`, loaded at runtime. Never inline prompts in Python code.
- Comments match blog voice: direct, no hype, no "revolutionary" or "seamless." If a comment sounds like marketing, rewrite it.

### Error Handling
- Schema validation failures MUST raise, never silently accept bad data.
- LLM parse failures in the evaluator are logged as warnings but don't abort the full document evaluation. One bad parse should not kill a batch.
- Missing files raise `FileNotFoundError` with descriptive messages.

### Testing
- All tests use mocked LLM responses. Zero real API calls in tests.
- Run: `uv run pytest tests/ -v`
- Minimum bar: all existing tests pass before any PR.

### Dependencies
- Managed with `uv`. `pyproject.toml` uses hatchling build backend.
- Install: `uv sync`
- Core deps: pydantic, anthropic, openai, python-dotenv, tiktoken.

## Domain: Technical Documentation Quality

The rubric domain is technical documentation review (API docs, tutorials, READMEs). NOT contact-center QA, NOT insurance, NOT call transcripts. This domain choice is deliberate — do not introduce call-center, insurance, or customer-agent language anywhere in the codebase.

Rules in `clean.csv` use these categories: Structure, Completeness, Code Quality, Accuracy, Compliance.

## Ground Truth Coupling

The 5 example documents in `examples/documents/` have `ground_truth` arrays referencing specific `rule_id` values (STRUCT-001, COMP-001, etc.) from `clean.csv`. The pre-compiled rubric in `examples/compiled/clean_compiled.json` has matching IDs. If you add or rename rules, you must update ground truth labels in all 5 document files.

## Common Tasks

### Add a new rule to the clean rubric
1. Add row to `examples/rubrics/clean.csv`
2. Add the rule to `examples/compiled/clean_compiled.json`
3. Add `ground_truth` entries for the new rule_id in all 5 document JSONs
4. Run tests: `uv run pytest tests/ -v`

### Add a new example document
1. Create `examples/documents/doc_NNN.json` with `doc_id`, `sections`, `metadata`, `ground_truth`
2. Ground truth must reference rule_ids from the clean rubric
3. Run eval harness to verify: `uv run python -m rubric_eval.eval examples/documents/ examples/compiled/clean_compiled.json --provider openai`
4. Generate pre-evaluated verdicts for the new doc so the harness can run without API key

### Change a prompt
1. Edit the `.md` file in `src/rubric_eval/prompts/`
2. Re-run the compiler or evaluator to test
3. Re-run eval harness to measure impact on accuracy
4. Update the sample output in README.md if metrics change

### Add a new LLM provider
1. Add a new class implementing `LLMClient` in `llm.py`
2. Add it to the `_PROVIDERS` dict
3. Document env vars in `.env.example`

## What NOT To Do

- Do not add insurance, call-center, contact-center, or customer-agent domain language. The domain is technical documentation.
- Do not add rule filtering or relevance routing to the evaluator. The brute force is intentional — it's the pattern to measure, not to optimize.
- Do not add agentic/tool-use evaluation. The agentic mode was deliberately stripped from this repo. See RegTriage for that pattern.
- Do not inline prompts in Python source. Prompts stay in `prompts/*.md`.
- Do not add streaming, retries, or caching to `llm.py`. This is a reference implementation, not production infrastructure.
- Do not make tests depend on real API calls.

## Running End-to-End

```bash
# Requires an API key in .env
uv run python -m rubric_eval.compiler examples/rubrics/clean.csv > compiled.json
uv run python -m rubric_eval.evaluator examples/documents/doc_001.json compiled.json
uv run python -m rubric_eval.eval examples/documents/ compiled.json

# Or use pre-compiled rubric with pre-evaluated verdicts (no API key needed)
uv run python -m rubric_eval.eval examples/documents/ examples/compiled/clean_compiled.json \
  --pre-evaluated examples/pre-evaluated/
```
