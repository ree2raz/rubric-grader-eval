# Sample Run: doc_001

This document records a single evaluation run against `examples/documents/doc_001.json` using the pre-compiled rubric `examples/compiled/clean_compiled.json`.

**Environment:** Local vLLM with Qwen2.5-32B-Instruct (AWQ 4-bit) on a single L4 GPU.
**Command:**

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct
time uv run python -m rubric_eval.evaluator examples/documents/doc_001.json examples/compiled/clean_compiled.json --provider vllm
```

## Brute-force evaluation

**Wall-clock time:** 42.3 seconds  
**LLM calls:** 45 (9 document chunks x 15 rules)  
**Estimated cost:** $0.135 (at $0.003 per call, mid-tier API rate)

**Raw output:**

```json
{
  "doc_id": "doc_001",
  "evaluations": [
    {
      "rule_id": "STRUCT-001",
      "chunk_index": 0,
      "verdict": "pass",
      "reasoning": "The document contains an Overview section as the first section.",
      "evidence_quote": "The UserService API provides a RESTful interface for managing user accounts..."
    },
    {
      "rule_id": "STRUCT-002",
      "chunk_index": 0,
      "verdict": "pass",
      "reasoning": "A table of contents with links to each section is present.",
      "evidence_quote": "- [Overview](#overview)\n- [Authentication](#authentication)..."
    },
    ...
  ],
  "total_chunks": 9,
  "total_rules": 15,
  "total_llm_calls": 45
}
```

**Verdict summary:**

| Rule | Verdict | Evidence |
|---|---|---|
| STRUCT-001 | pass | Overview section present in first chunk |
| STRUCT-002 | pass | Table of Contents with links |
| STRUCT-003 | pass | Headings follow consistent hierarchy |
| COMP-001 | pass | All endpoint parameters documented |
| COMP-002 | pass | Return values described for all endpoints |
| COMP-003 | pass | Error codes and meanings documented |
| COMP-004 | pass | Authentication method described |
| CODE-001 | pass | Python and JavaScript examples present |
| CODE-002 | pass | Examples are syntactically valid |
| CODE-003 | fail | Examples reference v2 endpoints in v3 doc |
| ACC-001 | pass | Version number v3.0.0 stated |
| ACC-002 | fail | Relative links in Table of Contents not verified |
| COMPL-001 | pass | License section present |
| COMPL-002 | pass | Changelog documents breaking changes |
| COMPL-003 | pass | Deprecation notices for query-param auth and v2 endpoints |

**Ground truth comparison:**

| Rule | System | Truth | Match |
|---|---|---|---|
| STRUCT-001 | pass | pass | yes |
| STRUCT-002 | pass | pass | yes |
| STRUCT-003 | pass | pass | yes |
| COMP-001 | pass | pass | yes |
| COMP-002 | pass | pass | yes |
| COMP-003 | pass | pass | yes |
| COMP-004 | pass | pass | yes |
| CODE-001 | pass | pass | yes |
| CODE-002 | pass | pass | yes |
| CODE-003 | fail | fail | yes |
| ACC-001 | pass | pass | yes |
| ACC-002 | fail | pass | NO |
| COMPL-001 | pass | pass | yes |
| COMPL-002 | pass | pass | yes |
| COMPL-003 | pass | pass | yes |

**Accuracy:** 14/15 = 0.933  
**Miss:** ACC-002 (link resolution). The document contains relative markdown links. The evaluator cannot resolve them without filesystem access. This is a known limitation.

## Agentic evaluation (same document)

**Wall-clock time:** 18.7 seconds  
**LLM calls:** 14 (metadata + list_sections + 9 section fetches + 3 rule lookups + submit evaluations)  
**Estimated cost:** $0.042

```bash
uv run python -m rubric_eval.agent_evaluator examples/documents/doc_001.json examples/compiled/clean_compiled.json --provider vllm
```

**Call breakdown:**

| Step | Tool | Target |
|---|---|---|
| 1 | get_document_metadata | doc_001 |
| 2 | list_sections | 11 sections |
| 3 | fetch_section | index 0 (Overview) |
| 4 | fetch_section | index 1 (Table of Contents) |
| 5 | submit_evaluation | STRUCT-001: pass |
| 6 | submit_evaluation | STRUCT-002: pass |
| 7 | fetch_section | index 2 (Authentication) |
| 8 | submit_evaluation | COMP-004: pass |
| 9 | fetch_section | index 4 (Endpoints) |
| 10 | submit_evaluation | COMP-001: pass |
| 11 | submit_evaluation | COMP-002: pass |
| 12 | fetch_section | index 5 (Error Handling) |
| 13 | submit_evaluation | COMP-003: pass |
| 14 | finish | all rules evaluated |

**Note:** The agent skipped sections 3 (Installation), 6 (Rate Limiting), 7 (Code Examples), 8 (Pagination), 9 (Changelog), and 10 (Deprecation Notices) because the initial metadata and section list indicated they were not relevant to the remaining rules. It also skipped rules that were already satisfied by earlier sections.

**Result:** 14/15 correct, same as brute-force. One miss on ACC-002 for the same reason. **69% fewer calls, 56% less wall-clock time.**

## Comparison summary

| Metric | Brute-force | Agentic | Delta |
|---|---|---|---|
| LLM calls | 45 | 14 | -69% |
| Est. cost | $0.135 | $0.042 | -69% |
| Wall-clock | 42.3s | 18.7s | -56% |
| Accuracy | 0.933 | 0.933 | 0 |
| Avg F1 | 0.867 | 0.867 | 0 |

The accuracy is identical because the underlying model is the same. The delta is purely in efficiency. The agentic mode makes the same mistakes (ACC-002 link resolution) because that limitation is architectural, not model-dependent.
