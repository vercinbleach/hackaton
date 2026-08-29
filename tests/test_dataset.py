from __future__ import annotations

from pathlib import Path

import pytest

from cala_fastpath_training.catalog import load_catalog
from cala_fastpath_training.compiler import compile_plan
from cala_fastpath_training.dataset import (
    grouped_split,
    to_pioneer_row,
    validate_examples,
    validate_pioneer_jsonl,
    write_jsonl,
)
from cala_fastpath_training.models import Plan
from cala_fastpath_training.seed_data import bootstrap_examples

ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT / "training" / "config" / "catalog.json")


def test_bootstrap_examples_are_valid() -> None:
    validate_examples(bootstrap_examples(), CATALOG)


def _example(example_id: str):
    return next(row for row in bootstrap_examples() if row.id == example_id)


def test_google_case_uses_flat_labels_and_semantic_entity() -> None:
    pioneer = to_pioneer_row(_example("seed-001"), CATALOG)

    assert pioneer.model_dump() == {
        "text": _example("seed-001").text,
        "labels": [
            "operation:knowledge_query",
            "root:companies",
            "filter:previous_job_eq",
            "return:founder",
        ],
        "entities": [
            {
                "text": "Google",
                "label": "founder_previous_employer_filter",
                "start": 45,
                "end": 51,
            }
        ],
    }


def test_retrieve_entity_has_root_returns_and_target_entity() -> None:
    pioneer = to_pioneer_row(_example("seed-013"), CATALOG)

    assert pioneer.labels == [
        "operation:retrieve_entity",
        "root:companies",
        "return:employee_count",
        "return:registered_address",
    ]
    assert [(entity.label, entity.text) for entity in pioneer.entities] == [
        ("target_entity", "OpenAI")
    ]


def test_unsupported_has_only_operation_and_reason_truths() -> None:
    pioneer = to_pioneer_row(_example("seed-018"), CATALOG)

    assert pioneer.labels == ["operation:unsupported", "reason:open_ended_explanation"]
    assert pioneer.entities == []


def test_limit_and_order_use_native_heads_and_result_limit_entity() -> None:
    pioneer = to_pioneer_row(_example("seed-011"), CATALOG)

    assert pioneer.labels == [
        "operation:knowledge_query",
        "root:startups",
        "filter:location_eq",
        "return:funding",
        "return:sector",
        "order_by:funding:desc",
    ]
    assert [(entity.label, entity.text) for entity in pioneer.entities] == [
        ("location_exact_filter", "Spain"),
        ("result_limit", "5"),
    ]


def test_pioneer_rows_have_no_legacy_fields(tmp_path: Path) -> None:
    rows = [to_pioneer_row(row, CATALOG) for row in bootstrap_examples()]
    output_path = tmp_path / "pioneer.jsonl"
    write_jsonl(output_path, rows)
    serialized = output_path.read_text(encoding="utf-8")
    entity_labels = {entity.label for row in rows for entity in row.entities}

    assert "plan_tags" not in serialized
    assert "previous_job_eq" not in entity_labels
    assert "target_entity" in entity_labels
    assert "result_limit" in entity_labels
    assert "json_structures" not in serialized
    assert '"input"' not in serialized
    assert '"output"' not in serialized
    assert "return:name" not in serialized
    assert validate_pioneer_jsonl(output_path, CATALOG) == {
        "valid": len(rows),
        "invalid": 0,
        "total": len(rows),
        "invalid_indices": [],
        "errors": [],
    }


def test_pioneer_schema_uses_classification_and_ner_without_structures() -> None:
    schema = CATALOG.inference_schema()

    assert set(schema) == {"classifications", "entities"}
    assert schema["entities"]["founder_previous_employer_filter"]
    assert "structures" not in schema


def test_grouped_split_has_no_group_leakage() -> None:
    split = grouped_split(bootstrap_examples(), seed=42)
    groups = [
        {row.group for row in split.train},
        {row.group for row in split.validation},
        {row.group for row in split.test},
    ]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]


def test_compile_plan_orders_modifiers() -> None:
    plan = Plan.model_validate(
        {
            "operation": "knowledge_query",
            "root": "startups",
            "filters": [
                {"kind": "location_eq", "mention": "Spain", "value": "Spain"},
                {"kind": "funding_gt", "mention": "10M", "value": "10M"},
            ],
            "order_by": "funding:desc",
            "limit": 5,
            "limit_mention": "5",
            "return": ["name", "funding"],
        }
    )
    assert compile_plan(plan, CATALOG) == (
        "startups.location=Spain.funding>10M.order_by=funding DESC.limit=5.return(name, funding)"
    )


def test_write_jsonl_rejects_symlink_destination(tmp_path: Path) -> None:
    """Test that write_jsonl refuses to write to a symlink destination."""
    target = tmp_path / "target.jsonl"
    target.write_text("original content", encoding="utf-8")

    symlink = tmp_path / "symlink.jsonl"
    try:
        symlink.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    examples = bootstrap_examples()[:1]  # Use just one example for the test

    with pytest.raises(ValueError, match="Refusing to write to symlink"):
        write_jsonl(symlink, examples)

    # Verify the target file was not modified
    assert target.read_text(encoding="utf-8") == "original content"


def test_write_jsonl_works_with_regular_file(tmp_path: Path) -> None:
    """Test that write_jsonl works correctly with regular files."""
    output = tmp_path / "output.jsonl"
    examples = bootstrap_examples()[:2]  # Use just two examples for the test

    write_jsonl(output, examples)

    assert output.exists()
    assert not output.is_symlink()

    # Verify the content is valid JSONL
    lines = output.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
