"""
Tests for the JSON schema and Pydantic model validation.

These tests verify that the schema enforces the contract: required fields
must be present, types must be correct, optional fields are truly optional,
and composite rules validate their sub_conditions structure.
"""

import pytest
from pydantic import ValidationError

from rubric_eval.models import Rule, SubCondition, CompiledRubric


class TestRuleValidation:
    """Test Rule model validation against the schema contract."""

    def test_valid_rule_passes(self):
        """A rule with all required fields validates successfully."""
        rule = Rule(
            rule_id="STRUCT-001",
            rule_name="Overview Section Present",
            description="The document must contain an overview section.",
            category="Structure",
            is_autofail=False,
            points=5,
        )
        assert rule.rule_id == "STRUCT-001"
        assert rule.points == 5
        assert rule.sub_conditions is None
        assert rule.example is None

    def test_missing_required_field_raises(self):
        """Omitting a required field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Rule(
                rule_id="STRUCT-001",
                rule_name="Overview Section Present",
                # description is missing
                category="Structure",
                is_autofail=False,
                points=5,
            )
        assert "description" in str(exc_info.value)

    def test_optional_example_is_optional(self):
        """The example field can be omitted or set to None."""
        rule_no_example = Rule(
            rule_id="TEST-001",
            rule_name="Test Rule",
            description="Test description.",
            category="Test",
            is_autofail=False,
            points=1,
        )
        assert rule_no_example.example is None

        rule_with_example = Rule(
            rule_id="TEST-002",
            rule_name="Test Rule",
            description="Test description.",
            example="This is an example.",
            category="Test",
            is_autofail=False,
            points=1,
        )
        assert rule_with_example.example == "This is an example."

    def test_sub_conditions_validates_composite(self):
        """Composite rules with sub_conditions validate correctly."""
        rule = Rule(
            rule_id="BC-001",
            rule_name="Quickstart AND API Reference",
            description="Both sections must be present.",
            category="Structure",
            is_autofail=False,
            points=5,
            sub_conditions=[
                SubCondition(condition="Quickstart section present", operator="AND"),
                SubCondition(condition="API reference section present", operator="AND"),
            ],
        )
        assert len(rule.sub_conditions) == 2
        assert rule.sub_conditions[0].operator == "AND"

    def test_invalid_sub_condition_operator_raises(self):
        """Sub-conditions with invalid operators raise ValidationError."""
        with pytest.raises(ValidationError):
            Rule(
                rule_id="BC-001",
                rule_name="Test",
                description="Test",
                category="Test",
                is_autofail=False,
                points=1,
                sub_conditions=[
                    SubCondition(condition="Test", operator="XOR"),
                ],
            )

    def test_negative_points_raises(self):
        """Points must be >= 0."""
        with pytest.raises(ValidationError):
            Rule(
                rule_id="TEST-001",
                rule_name="Test",
                description="Test",
                category="Test",
                is_autofail=False,
                points=-1,
            )


class TestCompiledRubric:
    """Test CompiledRubric model."""

    def test_compiled_rubric_from_rules(self):
        """CompiledRubric wraps a list of rules with metadata."""
        rules = [
            Rule(
                rule_id="TEST-001",
                rule_name="Test",
                description="Test",
                category="Test",
                is_autofail=False,
                points=1,
            )
        ]
        rubric = CompiledRubric(rules=rules, source_file="test.csv")
        assert len(rubric.rules) == 1
        assert rubric.source_file == "test.csv"
        assert rubric.compiled_at is not None
