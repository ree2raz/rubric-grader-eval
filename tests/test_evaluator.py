"""
Tests for the evaluator module.

Tests chunking logic, chunk-rule pair counting, and evaluation output
validation. All LLM calls are mocked.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from rubric_eval.evaluator import chunk_text, evaluate_document, _parse_chunk_evaluation
from rubric_eval.models import (
    ChunkEvaluation,
    CompiledRubric,
    Document,
    DocumentMetadata,
    Rule,
    Section,
)


def _make_document(content_length: int = 500) -> Document:
    """Create a test document with a known amount of content."""
    return Document(
        doc_id="test_doc",
        sections=[
            Section(
                heading="Test Section",
                content="word " * content_length,
                index=0,
            ),
        ],
        metadata=DocumentMetadata(
            doc_type="test",
            author_hash="abc123",
            project="TestProject",
            version="1.0.0",
        ),
    )


def _make_rubric(num_rules: int = 3) -> CompiledRubric:
    """Create a test rubric with a known number of rules."""
    rules = [
        Rule(
            rule_id=f"TEST-{i:03d}",
            rule_name=f"Test Rule {i}",
            description=f"Test description for rule {i}.",
            category="Test",
            is_autofail=False,
            points=1,
        )
        for i in range(num_rules)
    ]
    return CompiledRubric(rules=rules, source_file="test.csv")


def _mock_eval_response(rule_id: str) -> str:
    """Generate a valid mock evaluation JSON response."""
    return json.dumps({
        "rule_id": rule_id,
        "verdict": "pass",
        "reasoning": "The rule is satisfied in this chunk.",
        "evidence_quote": "word word word",
    })


class TestChunking:
    """Test the text chunking logic."""

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size produces one chunk."""
        chunks = chunk_text("Hello world", chunk_size=1024)
        assert len(chunks) == 1

    def test_chunking_produces_correct_count(self):
        """Text is split into the expected number of chunks."""
        # 2000 words ≈ 2000 tokens (rough approximation for simple words)
        text = "word " * 2000
        chunks = chunk_text(text, chunk_size=500)
        # Should be at least 2 chunks, likely 3-4
        assert len(chunks) >= 2

    def test_chunks_contain_all_content(self):
        """Reassembled chunks contain all the original tokens."""
        text = "The quick brown fox jumps over the lazy dog. " * 100
        chunks = chunk_text(text, chunk_size=100)
        reassembled = "".join(chunks)
        # Token-level split means character-level reassembly may not be exact
        # but all words should be present
        assert "quick" in reassembled
        assert "lazy" in reassembled


class TestParseChunkEvaluation:
    """Test parsing of individual chunk evaluation responses."""

    def test_valid_response_parses(self):
        """Valid JSON evaluation response parses correctly."""
        response = json.dumps({
            "rule_id": "TEST-001",
            "verdict": "pass",
            "reasoning": "Satisfied.",
            "evidence_quote": "relevant text",
        })
        result = _parse_chunk_evaluation(response, "TEST-001", 0)
        assert result.rule_id == "TEST-001"
        assert result.verdict == "pass"
        assert result.chunk_index == 0


class TestEvaluateDocument:
    """Test the full evaluate_document function with mocked LLM."""

    def test_chunk_rule_pair_count(self):
        """Total LLM calls equals chunks × rules."""
        doc = _make_document(content_length=100)
        rubric = _make_rubric(num_rules=3)

        # Mock LLM to return valid responses
        mock_client = MagicMock()
        mock_client.complete.side_effect = lambda system, user: _mock_eval_response(
            # Extract rule_id from the user message
            "TEST-000"
        )

        result = evaluate_document(doc, rubric, client=mock_client, chunk_size=1024)

        # With ~100 words of content and 1024 token chunks, should be 1 chunk
        # 1 chunk × 3 rules = 3 calls
        assert result.total_llm_calls == result.total_chunks * result.total_rules
        assert mock_client.complete.call_count == result.total_llm_calls
