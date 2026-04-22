# llm-rubric-eval

This is a reference implementation demonstrating evaluation patterns for LLM rubric grading. It is intentionally minimal: synthetic data, hand-labeled golden set, visible failure modes. Use it as a skeleton for thinking about how to build and measure rubric-driven LLM systems in your domain.

## What this is

A two-stage pipeline and eval harness:

1. **Compile**: Take a rubric in any format (clean CSV, composite-annotated spreadsheet, broken PDF-export) and extract structured rules via LLM.
2. **Evaluate**: Run every rule against every chunk of a document. Brute force. One LLM call per chunk-rule pair.
3. **Measure**: Compare system verdicts against hand-labeled ground truth. Per-category precision, recall, F1. Per-rule agreement table.

The eval harness is the point. The compiler and evaluator demonstrate the pattern. The harness tells you whether the pattern works.

## What this is not

- **Not a product.** Synthetic data, no optimizations, no production error handling.
- **Not the 2026 architecture.** The blog post describes an agentic rebuild with tool use, budget-aware routing, and eval-driven prompt optimization. This repo does none of that. For that pattern, see [RegTriage](https://github.com/ree2raz/reg-triage-openenv).
- **Not optimized.** The evaluator runs every rule on every chunk. A rule checking for an overview section evaluates the changelog. This is the part to fix first — the repo makes the cost visible so you can measure the improvement when you do.

## Quickstart

```bash
# Install dependencies
uv sync

# Set your API key (default provider: Anthropic)
export ANTHROPIC_API_KEY=your-key-here

# Compile a rubric
uv run python -m rubric_eval.compiler examples/rubrics/clean.csv > compiled_rubric.json

# Evaluate a single document
uv run python -m rubric_eval.evaluator examples/documents/doc_001.json compiled_rubric.json

# Run the eval harness against the golden set
uv run python -m rubric_eval.eval examples/documents/ compiled_rubric.json
```

Other providers:

```bash
# OpenAI
export OPENAI_API_KEY=your-key
uv run python -m rubric_eval.compiler examples/rubrics/clean.csv --provider openai

# Local vLLM
export VLLM_BASE_URL=http://localhost:8000/v1
uv run python -m rubric_eval.compiler examples/rubrics/clean.csv --provider vllm
```

## Architecture

**Stage 1 — Compilation.** The compiler reads a rubric file as raw text (not as a structured CSV — some of these files are not valid CSVs). It sends the raw content to an LLM with a prompt that handles three variance patterns: multi-condition rules that need expansion, boolean composites encoded in comment cells, and broken documents masquerading as spreadsheets. The LLM output is validated against a Pydantic schema. Invalid output raises immediately — it does not get silently accepted downstream.

**Stage 2 — Evaluation.** The evaluator concatenates document sections, chunks by token count (1024 tokens, configurable), and runs every rule against every chunk. One LLM call per pair. The prompt asks for a verdict (pass/fail), reasoning, and a verbatim evidence quote. Results are aggregated: if any chunk passes a rule, the document-level verdict for that rule is "pass."

## Eval Harness

The eval harness (`eval.py`) is the part worth studying. It loads documents with hand-labeled ground truth, runs the evaluator, and compares:

- **Per-category metrics**: Precision, recall, F1 for each rule category (Structure, Completeness, Code Quality, Accuracy, Compliance)
- **Overall accuracy**: Fraction of rule verdicts matching ground truth
- **Agreement table**: Per-rule, per-document comparison showing exactly where the system agrees and disagrees with human judgment

This implements the "micro golden set" pattern: a small, hand-labeled reference set that tells you whether your system is improving or regressing when you change prompts, models, or chunking strategies.

## Limitations

Four known failure modes, documented here because shipping with honest limitations is more useful than hiding them:

1. **Multi-condition rules spanning chunks.** A rule requiring multiple conditions (e.g., "overview AND changelog present") can only evaluate whether both appear in the same chunk. If they appear in different chunks, neither chunk evaluation sees the full picture. The system has no cross-chunk aggregation for composite rules.

2. **Linear scaling.** Evaluation time is proportional to document length × rule count. A 20-section document with 15 rules and 4 chunks generates 60 LLM calls. Longer documents or larger rubrics scale accordingly. No relevance filtering reduces this.

3. **Low-quality source text.** If the document text is garbled (OCR artifacts, encoding errors, truncated content), the LLM evaluates garbage and produces garbage output. No quality filter gates the input.

4. **No relevance routing.** Every rule runs on every chunk regardless of whether the chunk could possibly be relevant. A "license section present" rule evaluates the code examples section. This wastes compute and is the easiest failure mode to fix — an embedding-based router would cut inference load significantly.

## Background Reading

- [Automating Insurance Call-Center QA: What Worked, What Broke, and What I'd Rebuild](TODO_BLOG_URL) — the blog post that describes these rubric grading patterns in a production context
- [RegTriage](https://github.com/ree2raz/reg-triage-openenv) — an agentic environment for compliance auditing that implements the "what I'd rebuild" architecture

## License

Apache 2.0. See [LICENSE](./LICENSE).
