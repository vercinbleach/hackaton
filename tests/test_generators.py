from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from cala_fastpath_training.catalog import load_catalog
from cala_fastpath_training.generators import GLiNERBaseGenerator, load_skill
from cala_fastpath_training.openai_responses import OpenAIResponsesClient, plan_output_schema

ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT / "training" / "config" / "catalog.json")


def test_openai_responses_uses_strict_plan_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "gpt-5.6-luna"
        assert body["reasoning"] == {"effort": "high"}
        assert body["text"]["format"] == {
            "type": "json_schema",
            "name": "cala_query_plan",
            "strict": True,
            "schema": plan_output_schema(CATALOG),
        }
        output = {
            "operation": "knowledge_query",
            "root": "companies",
            "filters": [{"kind": "previous_job_eq", "mention": "Google", "value": "Google"}],
            "return": ["name", "founder"],
            "entity": None,
            "order_by": None,
            "limit": None,
            "limit_mention": None,
            "reason": None,
        }
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "model": "test-model",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(output)}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            },
        )

    with OpenAIResponsesClient("test-key", transport=httpx.MockTransport(handler)) as client:
        plan, _, usage = client.generate_plan(
            "compañías fundadas por ex empleados de Google",
            model="gpt-5.6-luna",
            reasoning_effort="high",
            instructions="test",
            catalog=CATALOG,
        )

    assert plan.return_fields == ["name", "founder"]
    assert usage == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}


def test_gliner_output_converts_to_shared_plan() -> None:
    generator = GLiNERBaseGenerator(model="unused", catalog=CATALOG)
    plan = generator._to_plan(
        {
            "plan_tags": [
                {"label": "root:companies", "confidence": 0.9},
                {"label": "return:founder", "confidence": 0.8},
            ]
        },
        {"previous_job_eq": [{"value": {"text": "Google", "confidence": 0.99}}]},
    )

    assert plan.operation == "knowledge_query"
    assert plan.root == "companies"
    assert plan.return_fields == ["founder"]
    assert plan.filters[0].value == "Google"


def test_project_skill_contains_projection_rule() -> None:
    skills_root = ROOT / "benchmark" / "skills"
    skill_path = skills_root / "cala-query" / "SKILL.md"
    content = load_skill(skill_path, skills_root)
    assert "only explicitly requested output properties" in content


def test_load_skill_rejects_absolute_path_outside_allowed_root(tmp_path: Path) -> None:
    """Reject absolute paths that point outside the allowed root."""
    outside = tmp_path / "outside.md"
    outside.write_text("sensitive data", encoding="utf-8")
    skills_root = tmp_path / "skills"
    skills_root.mkdir()

    with pytest.raises(ValueError, match="must stay within"):
        load_skill(outside, skills_root)


def test_load_skill_rejects_relative_path_traversal(tmp_path: Path) -> None:
    """Reject path traversal attempts using ../."""
    outside = tmp_path / "outside.md"
    outside.write_text("sensitive data", encoding="utf-8")
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    traversal = skills_root / ".." / "outside.md"

    with pytest.raises(ValueError, match="must stay within"):
        load_skill(traversal, skills_root)


def test_load_skill_rejects_symlink_to_outside_file(tmp_path: Path) -> None:
    """Reject symlinks that point to files outside the allowed root."""
    outside = tmp_path / "outside.md"
    outside.write_text("sensitive data", encoding="utf-8")
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    link = skills_root / "link.md"

    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic links or junctions"):
        load_skill(link, skills_root)


def test_load_skill_rejects_symlink_in_parent_directory(tmp_path: Path) -> None:
    """Reject paths where a parent directory is a symlink."""
    outside = tmp_path / "outside"
    outside.mkdir()
    skill_file = outside / "skill.md"
    skill_file.write_text("sensitive data", encoding="utf-8")

    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    linked_dir = skills_root / "linked"

    try:
        linked_dir.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic links or junctions"):
        load_skill(linked_dir / "skill.md", skills_root)


def test_load_skill_rejects_hardlink_to_outside_file(tmp_path: Path) -> None:
    """Reject hardlinks that alias files outside the allowed root."""
    outside = tmp_path / "outside.md"
    outside.write_text("sensitive data", encoding="utf-8")
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    hardlink = skills_root / "hardlink.md"

    try:
        os.link(outside, hardlink)
    except OSError as exc:
        pytest.skip(f"hardlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="multiple hard links"):
        load_skill(hardlink, skills_root)


def test_load_skill_rejects_directory(tmp_path: Path) -> None:
    """Reject directories instead of files."""
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    directory = skills_root / "subdir"
    directory.mkdir()

    with pytest.raises(ValueError, match="must be a file"):
        load_skill(directory, skills_root)


def test_load_skill_rejects_nonexistent_file(tmp_path: Path) -> None:
    """Reject paths that don't exist."""
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    nonexistent = skills_root / "nonexistent.md"

    with pytest.raises(ValueError, match="does not exist"):
        load_skill(nonexistent, skills_root)


def test_load_skill_accepts_valid_file_within_allowed_root(tmp_path: Path) -> None:
    """Accept a valid skill file within the allowed root."""
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    skill_file = skills_root / "valid.md"
    skill_file.write_text("valid skill content", encoding="utf-8")

    content = load_skill(skill_file, skills_root)
    assert content == "valid skill content"


def test_load_skill_strips_frontmatter(tmp_path: Path) -> None:
    """Strip YAML frontmatter from skill files."""
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    skill_file = skills_root / "with_frontmatter.md"
    skill_file.write_text("---\ntitle: Test\n---\nactual content", encoding="utf-8")

    content = load_skill(skill_file, skills_root)
    assert content == "actual content"


def test_load_skill_accepts_nested_subdirectories(tmp_path: Path) -> None:
    """Accept skill files in nested subdirectories within the allowed root."""
    skills_root = tmp_path / "skills"
    nested = skills_root / "category" / "subcategory"
    nested.mkdir(parents=True)
    skill_file = nested / "nested.md"
    skill_file.write_text("nested skill", encoding="utf-8")

    content = load_skill(skill_file, skills_root)
    assert content == "nested skill"
