"""
Tests for the rubric compiler.

All tests use mocked LLM responses. No real API calls.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from rubric_eval.compiler import compile_rubric, _parse_rules, _load_prompt


# A valid LLM response: two rules as JSON array
VALID_LLM_RESPONSE = json.dumps([
    {
        "rule_id": "STRUCT-001",
        "rule_name": "Overview Present",
        "description": "Document must have an overview section.",
        "category": "Structure",
        "is_autofail": False,
        "points": 5,
    },
    {
        "rule_id": "CODE-001",
        "rule_name": "Code Examples",
        "description": "At least one code example must be present.",
        "example": "```python\\nprint('hello')\\n```",
        "category": "Code Quality",
        "is_autofail": False,
        "points": 5,
    },
])

# LLM response with markdown fencing
FENCED_LLM_RESPONSE = f"```json\n{VALID_LLM_RESPONSE}\n```"

# Invalid: missing required field
INVALID_LLM_RESPONSE = json.dumps([
    {
        "rule_id": "STRUCT-001",
        "rule_name": "Overview Present",
        # missing: description, category, is_autofail, points
    },
])


class TestParseRules:
    """Test the _parse_rules function that validates LLM output."""

    def test_valid_json_parses(self):
        """Valid JSON array of rules parses into Rule objects."""
        rules = _parse_rules(VALID_LLM_RESPONSE)
        assert len(rules) == 2
        assert rules[0].rule_id == "STRUCT-001"
        assert rules[1].example is not None

    def test_fenced_json_parses(self):
        """Markdown-fenced JSON is handled correctly."""
        rules = _parse_rules(FENCED_LLM_RESPONSE)
        assert len(rules) == 2

    def test_invalid_json_raises(self):
        """Non-JSON output raises ValueError."""
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_rules("This is not JSON at all.")

    def test_invalid_rule_raises(self):
        """Rules missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            _parse_rules(INVALID_LLM_RESPONSE)


class TestCompileRubric:
    """Test the full compile_rubric function with mocked LLM."""

    def test_compile_with_mocked_client(self, tmp_path):
        """compile_rubric produces a valid CompiledRubric with a mocked LLM."""
        # Write a dummy CSV
        csv_file = tmp_path / "test_rubric.csv"
        csv_file.write_text("Rule ID,Name\nSTRUCT-001,Overview\n")

        # Mock the LLM client
        mock_client = MagicMock()
        mock_client.complete.return_value = VALID_LLM_RESPONSE

        rubric = compile_rubric(str(csv_file), client=mock_client)

        assert len(rubric.rules) == 2
        assert rubric.source_file == str(csv_file)
        mock_client.complete.assert_called_once()

    def test_compile_missing_file_raises(self):
        """compile_rubric raises FileNotFoundError for missing files."""
        mock_client = MagicMock()
        with pytest.raises(FileNotFoundError):
            compile_rubric("/nonexistent/path.csv", client=mock_client)


class TestPromptLoading:
    """Test that the prompt file loads correctly."""

    def test_prompt_file_exists_and_loads(self):
        """The compile_rubric.md prompt file exists and is non-empty."""
        prompt = _load_prompt()
        assert len(prompt) > 100
        assert "rule_id" in prompt  # Prompt should reference the schema
