from __future__ import annotations

from pathlib import Path

import pytest

from cala_fastpath_training.catalog import load_catalog
from cala_fastpath_training.compiler import compile_plan
from cala_fastpath_training.dataset import (
    grouped_split,
    to_pioneer_row,
    validate_examples,
    write_jsonl,
)
from cala_fastpath_training.models import Plan
from cala_fastpath_training.seed_data import bootstrap_examples

ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT / "training" / "config" / "catalog.json")


def test_bootstrap_examples_are_valid() -> None:
    validate_examples(bootstrap_examples(), CATALOG)


def test_pioneer_values_are_verbatim_spans() -> None:
    for row in bootstrap_examples():
        pioneer = to_pioneer_row(row)
        for structure in pioneer.json_structures or []:
            value = next(iter(structure.values()))["value"]
            assert value.casefold() in pioneer.text.casefold()


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
