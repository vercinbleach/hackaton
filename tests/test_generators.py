from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from cala_fastpath_training.catalog import load_catalog
from cala_fastpath_training.generators import GLiNERBaseGenerator, load_skill
from cala_fastpath_training.models import BenchmarkCase, GenerationRecord
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


class FakeExtractor:
    def __init__(self, classifications: dict, entities: dict) -> None:
        self.classifications = classifications
        self.entities = entities
        self.tasks: dict | None = None
        self.entity_types: dict | None = None

    def classify_text(self, _query: str, tasks: dict, **_kwargs: object) -> dict:
        self.tasks = tasks
        return self.classifications

    def extract_entities(self, _query: str, entity_types: dict, **_kwargs: object) -> dict:
        self.entity_types = entity_types
        return self.entities


def _classification_output(
    *,
    employer_intent_confidence: float = 0.9,
    explicit_returns: list[dict] | None = None,
) -> dict:
    labels = [
        {"label": "operation:knowledge_query", "confidence": 0.99},
        {"label": "root:companies", "confidence": 0.98},
        {"label": "filter:previous_job_eq", "confidence": employer_intent_confidence},
        {"label": "order_by:funding:desc", "confidence": 0.1},
        {"label": "reason:unsupported_property", "confidence": 0.1},
    ]
    labels.extend(explicit_returns or [])
    return {
        "plan_labels": labels,
    }


def _run_fake(
    *,
    employer: str | None,
    employer_intent_confidence: float = 0.9,
    explicit_returns: list[dict] | None = None,
) -> tuple[GenerationRecord, FakeExtractor]:
    entities = {"entities": {}}
    if employer is not None:
        entities = {
            "entities": {
                "founder_previous_employer_filter": [
                    {"text": employer, "confidence": 0.99}
                ]
            }
        }
    extractor = FakeExtractor(
        _classification_output(
            employer_intent_confidence=employer_intent_confidence,
            explicit_returns=explicit_returns,
        ),
        entities,
    )
    generator = GLiNERBaseGenerator(model="unused", catalog=CATALOG)
    generator._extractor = extractor
    record = generator.generate(
        BenchmarkCase(
            id="test",
            query=f"Empresas fundadas por exempleados de {employer or 'Google'}",
        )
    )
    return record, extractor


def test_gliner_implicit_projection_adds_name() -> None:
    record, extractor = _run_fake(employer="Google")

    assert record.error is None
    assert record.decision == "accepted"
    assert record.plan is not None
    assert record.plan.return_fields == ["name"]
    assert record.plan.filters[0].value == "Google"
    assert extractor.tasks is not None
    assert set(extractor.tasks) == {"plan_labels"}
    assert extractor.tasks["plan_labels"]["multi_label"] is True
    assert "return:name" not in extractor.tasks["plan_labels"]["labels"]
    assert all(task["cls_threshold"] == 0.5 for task in extractor.tasks.values())
    assert extractor.entity_types is not None
    assert extractor.entity_types["founder_previous_employer_filter"]
    assert "target_entity" in extractor.entity_types


def test_gliner_explicit_founder_projection_is_added() -> None:
    record, _ = _run_fake(
        employer="Google",
        explicit_returns=[{"label": "return:founder", "confidence": 0.91}],
    )

    assert record.error is None
    assert record.plan is not None
    assert record.plan.return_fields == ["name", "founder"]


def test_gliner_filter_intent_without_employer_is_rejected() -> None:
    record, _ = _run_fake(employer=None)

    assert record.plan is None
    assert record.cala_query is None
    assert record.decision == "abstained"
    assert record.abstention_reason == (
        "filter label/extraction mismatch: missing=['previous_job_eq'], extra=[]"
    )


def test_gliner_extracts_microsoft_as_the_same_span_role() -> None:
    record, _ = _run_fake(employer="Microsoft")

    assert record.error is None
    assert record.plan is not None
    assert record.plan.filters[0].kind == "previous_job_eq"
    assert record.plan.filters[0].value == "Microsoft"


def test_gliner_filter_structure_without_label_is_rejected() -> None:
    record, _ = _run_fake(employer="Google", employer_intent_confidence=0.49)

    assert record.plan is None
    assert record.decision == "abstained"
    assert record.abstention_reason == (
        "filter label/extraction mismatch: missing=[], extra=['previous_job_eq']"
    )


def test_gliner_rejects_two_values_for_the_same_entity_type() -> None:
    extractor = FakeExtractor(
        _classification_output(),
        {
            "entities": {
                "founder_previous_employer_filter": [
                    {"text": "Google", "confidence": 0.99},
                    {"text": "Microsoft", "confidence": 0.98},
                ]
            }
        },
    )
    generator = GLiNERBaseGenerator(model="unused", catalog=CATALOG)
    generator._extractor = extractor

    record = generator.generate(
        BenchmarkCase(id="duplicate", query="Empresas creadas por ex Google y Microsoft")
    )

    assert record.decision == "abstained"
    assert record.abstention_reason == (
        "filter intent 'previous_job_eq' requires one extracted employer"
    )


def test_gliner_accepts_flat_pioneer_entity_records() -> None:
    extractor = FakeExtractor(
        _classification_output(),
        {
            "entities": [
                {
                    "text": "Google",
                    "label": "founder_previous_employer_filter",
                    "confidence": 0.99,
                    "start": 28,
                    "end": 34,
                }
            ]
        },
    )
    generator = GLiNERBaseGenerator(model="unused", catalog=CATALOG)
    generator._extractor = extractor

    record = generator.generate(
        BenchmarkCase(id="flat", query="Empresas creadas por antiguos Google employees")
    )

    assert record.decision == "accepted"
    assert record.plan is not None
    assert record.plan.filters[0].mention == "Google"


def test_gliner_ignores_top_label_below_threshold() -> None:
    record, _ = _run_fake(
        employer="Google",
        explicit_returns=[{"label": "return:founder", "confidence": 0.49}],
    )

    assert record.decision == "accepted"
    assert record.plan is not None
    assert record.plan.return_fields == ["name"]
    assert record.plan.order_by is None


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
