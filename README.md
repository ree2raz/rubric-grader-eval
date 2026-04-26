# rubric-grader-eval

A reference pattern for compiling unstructured rubrics into machine-readable schemas, then evaluating documents against them with golden-set ground truth.

## What this is

Two stages, plus a measurement harness:

1. **Compile**: Take a rubric in any format — clean spreadsheet, composite-annotated notes, broken PDF-export — and extract structured rules via LLM. The compiler is the artifact. It is the hard part of the problem.
2. **Evaluate**: Brute-force. Every rule against every chunk of the document. One LLM call per pair. Deliberately simple — it demonstrates that the compiled rubric works end-to-end.
3. **Measure**: Compare system verdicts against hand-labeled ground truth. Per-category precision, recall, F1. Per-rule agreement tables. The eval harness proves the compiler worked.

The compiler is the point. Rubrics live in spreadsheets, PDFs, and SMEs' heads. No standard format. No API. The hard work is getting them into a machine-readable schema that captures composite logic and is validated on output. The evaluator and harness tell you whether that compilation produced rules that can be evaluated correctly.

## What this is not

- **Not a product.** Synthetic data, no optimizations, no production error handling.
- **Not an agentic system.** This is the brute-force baseline. For the agentic rebuild with budget-aware routing and eval-driven prompt optimization, see [RegTriage](https://github.com/ree2raz/RegTriage-OpenEnv).
- **Not optimized.** The evaluator is intentionally inefficient — every rule runs on every chunk so you can measure the cost of this approach.

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
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=Qwen/Qwen3-14B-AWQ

# Compile a rubric (calls LLM to extract rules from CSV)
uv run python -m rubric_eval.compiler examples/rubrics/clean.csv --provider vllm > compiled_rubric.json

# Evaluate a single document
uv run python -m rubric_eval.evaluator examples/documents/doc_001.json compiled_rubric.json --provider vllm

# Run the eval harness against the golden set
uv run python -m rubric_eval.eval examples/documents/ compiled_rubric.json --provider vllm

# Pure metrics mode: no LLM calls, just ground-truth comparison using pre-computed verdicts
uv run python -m rubric_eval.eval examples/documents/ examples/compiled/clean_compiled.json \
  --pre-evaluated examples/pre-evaluated/
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

## The compiler: three variance cases

Rubric compilation is where most projects fail. The input is not structured data. It is a spreadsheet with merged cells, a PDF scan of a printed table, or a CSV exported from a document that was never valid CSV. The compiler handles all three — one prompt, one LLM call, validated Pydantic output.

### 1. Clean CSV

The easy case. A well-structured spreadsheet with consistent columns, one rule per row.

**Input** (`examples/rubrics/clean.csv`):
```csv
rule_id,rule_name,category,description,is_autofail,points
STRUCT-001,Overview Section Present,Structure,The document must contain an Overview section as the first section.,false,5
COMP-001,Parameter Documentation,Completeness,All API endpoint parameters must be documented with type description and example.,false,10
...
```

**Output** (`examples/compiled/clean_compiled.json`):
```json
{
  "rules": [
    {
      "rule_id": "STRUCT-001",
      "rule_name": "Overview Section Present",
      "category": "Structure",
      "description": "The document must contain an Overview section as the first section.",
      "is_autofail": false,
      "points": 5,
      "sub_conditions": null
    }
  ]
}
```

15 rules across 5 categories. Standard extraction. Nothing clever needed.

### 2. Boolean composites

Rules with AND/OR logic hidden in a "Notes" column. The compiler must detect composite conditions and emit structured `sub_conditions` rather than splitting them into independent rules — splitting would double-count failures.

**Input** (`examples/rubrics/boolean_composite.csv`):
```csv
rule_id,rule_name,category,description,is_autofail,points,Notes
BC-001,Quickstart AND API Reference,Completeness,Document must include both a quickstart and a full API reference.,false,10,"AND: requires quickstart guide + full API reference"
BC-002,Install OR Setup,Completeness,Document must include either installation or setup instructions.,false,5,"OR: either installation or setup section"
...
```

**Output** (`examples/compiled/boolean_composite_compiled.json`):
```json
{
  "rule_id": "BC-001",
  "rule_name": "Quickstart AND API Reference",
  "category": "Completeness",
  "sub_conditions": [
    {"condition": "Quickstart guide present", "operator": "AND"},
    {"condition": "Full API reference section present", "operator": "AND"}
  ]
}
```

12 rules, 7 with composite `sub_conditions`. The evaluator reads these and evaluates each sub-condition against the chunk, combining via the operator.

### 3. Document masquerade

The hard case. A PDF exported to CSV — header rows appear mid-file, rules span multiple lines, section headings interleave with data rows. Column counts are inconsistent. You cannot parse this as structured CSV. The compiler reads it as raw text and reconstructs rules by content patterns.

**Input** (`examples/rubrics/document_masquerade.csv`) — a 50-line document with:
- Prose introduction paragraphs
- Section headings (`# Structure`, `# Completeness`, `# Code Quality`)
- Rules in semicolon-separated rows with variable column counts
- A trailing "Compliance & Accuracy" section with autofail annotations

**Output** (`examples/compiled/document_masquerade_compiled.json`) — 13 rules extracted:
```json
[
  {"rule_id": "S-1", "category": "Structure", "rule_name": "Overview Section Present"},
  {"rule_id": "S-2", "category": "Structure", "rule_name": "Table of Contents"},
  {"rule_id": "S-3", "category": "Structure", "rule_name": "Heading Hierarchy"},
  {"rule_id": "F-1", "is_autofail": true, "rule_name": "No Broken Links"},
  ...
]
```

Three rules marked `is_autofail: true` because the source text flagged them as mandatory. Categories inferred from section headings, not column values.

The compilation prompt lives in `src/rubric_eval/prompts/compile_rubric.md`. It is the most carefully tuned artifact in the repo.

## Evaluator

Brute-force. The document is chunked by 1024 tokens (tiktoken, no overlap). For each chunk×rule pair, one LLM call. No filtering. No relevance routing. This is the pattern to measure, not to optimize.

```bash
uv run python -m rubric_eval.evaluator examples/documents/doc_001.json examples/compiled/clean_compiled.json --provider vllm
```

Output is a `DocumentEvaluation`: a list of per-chunk per-rule verdicts with verbatim evidence quotes.

## Eval harness

Loads golden-set documents (each with hand-labeled `ground_truth` arrays), runs the evaluator, and prints per-category macro-F1 (averaged across both pass and fail classes) plus a per-rule agreement table.

Two modes:

```bash
# With LLM: run the evaluator against all docs, then compare
uv run python -m rubric_eval.eval examples/documents/ examples/compiled/clean_compiled.json --provider vllm

# Without LLM: load pre-computed verdicts, just run the comparison
uv run python -m rubric_eval.eval examples/documents/ examples/compiled/clean_compiled.json \
  --pre-evaluated examples/pre-evaluated/
```

The `--pre-evaluated` flag loads previously-saved `DocumentEvaluation` JSONs from disk. No LLM calls, no API key needed. The harness becomes a pure metrics calculator.

## Sample Output

Running the eval harness on all five example documents produces:

```
======================================================================
EVAL HARNESS RESULTS
======================================================================

Document: doc_001 | Calls: 45 | Accuracy: 0.933 | Avg F1: 0.480

Category                   MacroF1   FailF1   PassF1   TP   FP   FN
------------------------------------------------------------
Accuracy                     0.500    0.000    1.000    0    0    0
Code Quality                 0.500    0.000    1.000    0    0    0
Completeness                 0.500    0.000    1.000    0    0    0
Compliance                   0.400    0.000    0.800    0    1    0
Structure                    0.500    0.000    1.000    0    0    0

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

**MacroF1** is the key number — it averages F1 across both classes (pass and fail). A category where all rules pass and the system agrees gets a high PassF1 but FailF1 0.0 (no failures to detect). A category with mixed verdicts shows meaningful FailF1 and PassF1, and the macro average surfaces the balance.

The numbers are not good. They are honest. Every degraded category maps to a known failure mode documented below:

- **Structure FailF1 0.0, PassF1 1.0**: The system correctly confirms passes but never detects failures because the rules that fail (heading hierarchy) require cross-chunk reasoning. The evaluator has no cross-chunk memory. This is the multi-condition-spanning-chunks failure mode.

- **Code Quality FailF1 0.0, PassF1 0.8-1.0**: Syntax-validation rules (CODE-002) depend on the LLM's ability to parse code examples. Some examples are pseudo-code or shell commands that do not have strict syntax. The LLM is conservative and marks them as failing. This is the low-quality-source-text failure mode applied to code.

- **Accuracy FailF1 0.0**: Link-resolution rules (ACC-002) and version-number rules (ACC-001) fail when links are broken or versions are implicit. Both are known limitations of evaluating static documents without external lookups.

The Completeness and Compliance categories score consistently because their rules (parameter documentation, error handling, license presence) are locally verifiable within a single chunk. The gap between high and low categories is exactly the gap between "can be verified locally" and "requires cross-section or external context."

## Architecture

```
CSV rubric → [compiler.py] → compiled JSON rules → [evaluator.py] → chunk×rule verdicts → [eval.py] → metrics
```

- **compiler.py**: Reads rubric as raw text, sends to LLM, validates output via Pydantic. Handles three variance cases in one prompt. The most carefully tuned component.
- **evaluator.py**: Brute-force. Chunks document by 1024 tokens. Every rule × every chunk = one LLM call. Intentionally inefficient.
- **eval.py**: Golden-set comparison. Loads documents with `ground_truth` labels, runs evaluator (or loads pre-computed verdicts), computes per-category macro-averaged F1 and per-rule agreement tables.
- **llm.py**: Provider factory. Anthropic, OpenAI, vLLM. Unified `complete(system, user) -> str` interface.
- **models.py**: Pydantic v2 models. All LLM output is validated here. Invalid output raises.

## Limitations

Four known failure modes, documented here because shipping with honest limitations is more useful than hiding them:

1. **Multi-condition rules spanning chunks.** A rule requiring multiple conditions can only evaluate whether they appear in the same chunk. No cross-chunk aggregation. Composite rules with `sub_conditions` are evaluated per-chunk — if condition A appears in chunk 1 and condition B in chunk 3, an AND composite will fail both chunks.

2. **Linear scaling.** Brute-force evaluation time is proportional to `document length × rule count`. With 25 rules and 5-10 chunks per document, that's 125-250 LLM calls per document. Fine for batch evaluation. Expensive at scale.

3. **Low-quality source text.** OCR artifacts, encoding errors, or truncated content produce garbage output. No quality filter gates the input. The evaluator trusts that text is text.

4. **No relevance routing.** Every rule runs on every chunk regardless of relevance. A "license section present" rule evaluates the code examples section. The cost is visible in the Calls column of the eval output. This is the first thing to optimize in a rebuild.

## See also

- [RegTriage](https://github.com/ree2raz/RegTriage-OpenEnv) — an agentic compliance auditing environment. Implements the budget-aware, tool-mediated architecture that replaces brute-force evaluation.
- [audited-tool-mcp](https://github.com/ree2raz/audited-tool-mcp) — MCP server for auditable tool execution with structured logging and compliance traces.

## License

Apache 2.0. See [LICENSE](./LICENSE).
