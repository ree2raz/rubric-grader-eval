"""
Pydantic models for rubric compilation, evaluation, and golden-set comparison.

These models enforce the schema at every boundary: LLM output gets validated
on ingestion, not downstream. If the LLM produces invalid JSON, it fails here
with a clear error — not silently in the evaluator.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SubCondition(BaseModel):
    """One part of a composite rule. Combined via AND/OR with siblings."""

    condition: str = Field(description="What this sub-condition checks.")
    operator: Literal["AND", "OR"] = Field(
        description="How this combines with other sub-conditions."
    )


class Rule(BaseModel):
    """A single evaluable rule extracted from a rubric.

    Maps directly to the JSON schema in schema/rule_schema.json.
    Every rule the compiler emits must pass this model's validation.
    """

    rule_id: str = Field(description="Unique identifier, e.g. 'STRUCT-001'.")
    rule_name: str = Field(description="Short human-readable name.")
    description: str = Field(description="Full evaluation criteria.")
    example: str | None = Field(
        default=None, description="Optional example of pass/fail."
    )
    category: str = Field(description="Grouping category.")
    is_autofail: bool = Field(
        description="True if failing this rule fails the entire evaluation."
    )
    points: int = Field(ge=0, description="Point value.")
    sub_conditions: list[SubCondition] | None = Field(
        default=None,
        description="For composite rules only.",
    )


class CompiledRubric(BaseModel):
    """Output of the compiler: a validated set of rules ready for evaluation."""

    rules: list[Rule]
    source_file: str = Field(description="Path to the source CSV/document.")
    compiled_at: datetime = Field(default_factory=datetime.now)


class ChunkEvaluation(BaseModel):
    """Result of evaluating one rule against one chunk of text."""

    rule_id: str
    chunk_index: int = Field(ge=0)
    verdict: Literal["pass", "fail"]
    reasoning: str = Field(description="Why the rule passed or failed in this chunk.")
    evidence_quote: str = Field(
        description="Verbatim text from the chunk supporting the verdict."
    )


class DocumentEvaluation(BaseModel):
    """Full evaluation results for one document against a rubric."""

    doc_id: str
    evaluations: list[ChunkEvaluation]
    total_chunks: int = Field(ge=1)
    total_rules: int = Field(ge=1)
    total_llm_calls: int = Field(ge=0)


class GroundTruthEntry(BaseModel):
    """Hand-labeled verdict for one rule on one document.

    Used by the eval harness to compare system output against human judgment.
    """

    rule_id: str
    verdict: Literal["pass", "fail"]
    evidence_text: str = Field(
        description="Text from the document that justifies the verdict."
    )
    severity: Literal["critical", "major", "minor"]


class Section(BaseModel):
    """One section of a technical document."""

    heading: str
    content: str
    index: int = Field(ge=0)


class DocumentMetadata(BaseModel):
    """Metadata about a document under evaluation."""

    doc_type: str = Field(description="E.g. 'api_reference', 'tutorial', 'readme'.")
    author_hash: str = Field(description="Anonymized author identifier.")
    project: str
    version: str


class Document(BaseModel):
    """A document to evaluate, with optional ground truth labels."""

    doc_id: str
    sections: list[Section]
    metadata: DocumentMetadata
    ground_truth: list[GroundTruthEntry] | None = Field(
        default=None,
        description="Hand-labeled verdicts. Present only in golden-set documents.",
    )
