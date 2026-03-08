"""Tests for prompt templates."""
from petro_mcp.prompts.templates import get_template, list_templates, TEMPLATES


class TestGetTemplate:
    def test_get_existing_template(self):
        t = get_template("analyze_decline")
        assert t is not None
        assert "template" in t
        assert "description" in t

    def test_get_nonexistent_template(self):
        assert get_template("nonexistent") is None

    def test_all_templates_have_required_keys(self):
        for name, tmpl in TEMPLATES.items():
            assert "name" in tmpl, f"{name} missing 'name'"
            assert "description" in tmpl, f"{name} missing 'description'"
            assert "template" in tmpl, f"{name} missing 'template'"


class TestListTemplates:
    def test_list_templates_returns_all(self):
        result = list_templates()
        assert len(result) == len(TEMPLATES)

    def test_list_templates_format(self):
        result = list_templates()
        for item in result:
            assert "name" in item
            assert "description" in item
