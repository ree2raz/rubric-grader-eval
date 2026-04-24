# llm-rubric-eval

A reference implementation demonstrating how to compile rubrics from unstructured documents and evaluate them with LLMs. The compiler is the hard part. The eval harness measures whether the compiler worked.

## What this is

A three-stage pipeline:

1. **Compile**: Take a rubric in any format (clean CSV, composite-annotated spreadsheet, broken PDF-export masquerading as a spreadsheet) and extract structured rules via LLM. This is the most carefully tuned component.
2. **Evaluate**: Run rules against documents. Two modes: brute-force (every rule against every chunk) and agentic (targeted tool use, the agent decides what to read).
3. **Measure**: Compare system verdicts against hand-labeled ground truth. Per-category precision, recall, F1. Side-by-side brute vs agent comparison.

The compiler is the point. Rubrics live in spreadsheets, PDFs, and SME heads. No standard format. No API. The hard work is getting them into a machine-readable schema that captures composite logic. The evaluator and harness tell you whether that compilation produced rules that can be evaluated correctly.

## What this is not

- **Not a product.** Synthetic data, no optimizations, no production error handling.
- **Not the 2026 architecture.** The blog post describes an agentic rebuild with budget-aware routing and eval-driven prompt optimization. For that pattern, see [RegTriage](https://github.com/ree2raz/RegTriage-OpenEnv).
- **Not optimized.** The brute-force evaluator is intentionally inefficient. It runs every rule on every chunk so you can measure the delta when you switch to the agentic mode.

## Quickstart (local vLLM)

Self-hosted inference is the default path. Cloud providers work too, but the architecture assumes on-prem deployment.

```bash
# Install dependencies
uv sync

# Start a local vLLM server with a mid-size model
# Example: Qwen3-14B-Instruct-AWQ at 4-bit, easily fits on a single 24GB GPU (A10G, L4, RTX 3090)
vllm serve Qwen/Qwen3-14B-AWQ \
  --quantization awq \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 32 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000

# In another terminal, set the endpoint
export VLLM_BASE_URL=http://localhost:8000/v1 # or remote GPU hostname
export VLLM_MODEL=Qwen/Qwen3-14B-AWQ

# Compile a rubric (calls LLM to extract rules from CSV)
uv run python -m rubric_eval.compiler examples/rubrics/clean.csv --provider vllm > compiled_rubric.json

# Or use the included pre-compiled rubric (no API key needed for eval)
# cp examples/compiled/clean_compiled.json compiled_rubric.json

# Evaluate a single document (brute-force mode)
uv run python -m rubric_eval.evaluator examples/documents/doc_001.json compiled_rubric.json --provider vllm

# Run the eval harness against the golden set
uv run python -m rubric_eval.eval examples/documents/ compiled_rubric.json --provider vllm

# Compare brute-force vs agentic evaluation
uv run python -m rubric_eval.eval examples/documents/ compiled_rubric.json --provider vllm --mode compare
```

### Cloud providers (fallback)

```bash
# Anthropic
export ANTHROPIC_API_KEY=***
uv run python -m rubric_eval.compiler examples/rubrics/clean.csv --provider anthropic

# OpenAI
export OPENAI_API_KEY=***
uv run python -m rubric_eval.compiler examples/rubrics/clean.csv --provider openai
```

## Compiler: the hard part

Rubric compilation is where most projects fail. The input is not structured data. It is a spreadsheet with merged cells, a PDF scan of a printed table, or a CSV exported from a document that was never valid CSV. The compiler handles three variance patterns:

- **Clean CSVs**: Well-formed rubrics with standard columns. Extracted directly.
- **Boolean composites**: Rules encoded in comment cells with AND/OR logic. Expanded into structured sub-conditions.
- **Document masquerades**: PDF exports or broken documents pretending to be spreadsheets. Parsed as raw text, then extracted.

The output is validated Pydantic JSON with rule identity, pass/fail criteria, severity semantics, and composite logic preserved. Invalid output raises immediately. It does not get silently accepted downstream.

The compilation prompt lives in `src/rubric_eval/prompts/compile_rubric.md`. It is the most carefully tuned artifact in the repo.

## Evaluator: two modes

### Brute-force mode

Every rule runs against every chunk of the document. One LLM call per pair. Predictable, correct, and wasteful. This is the baseline you ship first.

### Agentic mode

A single agent with targeted tools decides what to read. Tools: `get_document_metadata`, `list_sections`, `fetch_section`, `get_rule`, `submit_evaluation`. The agent budgets its tool calls and stops when all rules are evaluated. This is the rebuild you measure against.

Run the comparison:

```bash
uv run python -m rubric_eval.eval examples/documents/ compiled_rubric.json --mode compare
```

This produces a side-by-side table: calls, estimated cost, accuracy, and per-category F1 for both modes on the same documents.

## Sample Output

Running the eval harness on all five example documents produces:

```
======================================================================
EVAL HARNESS RESULTS
======================================================================

Document: doc_001 | Calls: 45 | Accuracy: 0.800 | Avg F1: 0.600

Category                    Prec    Rec     F1   TP   FP   FN
------------------------------------------------------------
Accuracy                   0.200  0.500  0.286    1    4    1
Code Quality               1.000  0.500  0.667    1    0    1
Completeness               0.786  1.000  0.880   11    3    0
Compliance                 0.889  1.000  0.941    8    1    0
Structure                  0.286  1.000  0.444    2    5    0

Doc          Rule           System   Truth    Match
--------------------------------------------------
doc_001      STRUCT-001     pass     pass     yes
doc_001      COMP-001       pass     pass     yes
doc_001      ACC-002        fail     pass     NO
doc_001      COMPL-003      fail     pass     NO
...

======================================================================
```

### Reading the metrics

The numbers are not good. They are honest. Every degraded category maps to a known failure mode that was documented before the first eval run, not discovered after:

- **Accuracy F1 0.286**: Two failure modes. First, link-resolution rules (ACC-002) fail when links in the document are broken or relative. Second, version-number rules (ACC-001) fail when the document does not explicitly state a version. Both are known limitations of evaluating static documents without external lookups.

- **Structure F1 0.444**: Heading-hierarchy rules (STRUCT-003) are evaluated per-chunk. A chunk may contain a heading jump that is resolved in a later chunk. The brute-force evaluator has no cross-chunk memory. This is the multi-condition-spanning-chunks failure mode.

- **Code Quality F1 0.667**: Syntax-validation rules (CODE-002) depend on the LLM's ability to parse code examples. Some examples are pseudo-code or shell commands that do not have strict syntax. The LLM is conservative and marks them as failing. This is the low-quality-source-text failure mode applied to code.

The Completeness and Compliance categories score well because their rules (parameter documentation, error handling, license presence) are locally verifiable within a single chunk. The gap between high and low categories is exactly the gap between "can be verified locally" and "requires cross-section or external context."

A full recorded run with per-document breakdown is in [`docs/sample_run.md`](./docs/sample_run.md).

## Architecture

```
CSV rubric → [compiler.py] → compiled JSON rules
                                      ↓
                         ┌────────────┴────────────┐
                         ↓                         ↓
              [evaluator.py: brute]      [agent_evaluator.py: targeted]
                         ↓                         ↓
                         └────────────┬────────────┘
                                      ↓
                              [eval.py: compare]
                                      ↓
                         per-category precision/recall/F1
```

- **compiler.py**: Reads rubric as raw text, sends to LLM, validates output via Pydantic. Handles three variance cases.
- **evaluator.py**: Brute-force. Chunks document by 1024 tokens. Every rule x every chunk = one LLM call each.
- **agent_evaluator.py**: Agentic. Single agent loop with tool budget. Decides which sections to read.
- **eval.py**: Loads golden-set documents, runs evaluator(s), compares to ground truth, prints metrics.
- **llm.py**: Provider factory. Anthropic, OpenAI, vLLM. Unified interface.
- **models.py**: Pydantic v2 models. All LLM output is validated here.

## Brute-force vs agentic: the architectural contrast

| Dimension | Brute-force (rubric-grader-eval) | Agentic (RegTriage) |
|---|---|---|
| Reads per document | Every section, every rule | Targeted: agent decides |
| LLM calls per doc (15 rules, ~5 sections) | ~75 | ~12-20 |
| Cross-section memory | None (per-chunk evaluation) | Yes (agent retains state) |
| Composite rule handling | Per-chunk only | Can aggregate across sections |
| Failure mode: relevance | Evaluates irrelevant sections | Skips irrelevant sections |
| Failure mode: debugging | Easy (deterministic loop) | Hard (conditional decisions) |
| When to use | Baseline, production ship | Optimization, research |

The brute-force pattern is what you ship first. The agentic pattern is what you rebuild toward. Both are in this repo. The comparison mode measures the delta.

## Limitations

Four known failure modes, documented here because shipping with honest limitations is more useful than hiding them:

1. **Multi-condition rules spanning chunks.** A rule requiring multiple conditions can only evaluate whether they appear in the same chunk. No cross-chunk aggregation.

2. **Linear scaling.** Brute-force evaluation time is proportional to document length x rule count. The agentic mode reduces this but adds complexity.

3. **Low-quality source text.** OCR artifacts, encoding errors, or truncated content produce garbage output. No quality filter gates the input.

4. **No relevance routing in brute mode.** Every rule runs on every chunk regardless of relevance. The agentic mode fixes this but introduces its own failure modes (wrong section selection, budget exhaustion before all rules are evaluated).

## Background Reading

- [Automating Insurance Call-Center QA: What Worked, What Broke, and What I'd Rebuild](https://www.rituraj.info/posts/insurance-qa-llm-scorecard-pattern/) — the blog post that describes these rubric grading patterns in a production context
- [RegTriage](https://github.com/ree2raz/RegTriage-OpenEnv) — an agentic environment for compliance auditing that implements the "what I'd rebuild" architecture

## License

Apache 2.0. See [LICENSE](./LICENSE).
