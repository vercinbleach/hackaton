from __future__ import annotations

from pathlib import Path

import pytest

from cala_fastpath_training.catalog import load_catalog
from cala_fastpath_training.compiler import PlanCompilationError, compile_plan
from cala_fastpath_training.models import Plan

ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT / "training" / "config" / "catalog.json")


def make_plan(**updates: object) -> Plan:
    payload: dict[str, object] = {
        "operation": "knowledge_query",
        "root": "companies",
        "filters": [
            {"kind": "location_eq", "mention": "Spain", "value": "Spain"},
        ],
        "return": ["name"],
    }
    payload.update(updates)
    return Plan.model_validate(payload)


def test_adds_name_projection_by_default() -> None:
    plan = make_plan(**{"return": []})

    assert compile_plan(plan, CATALOG) == "companies.location=Spain.return(name)"


def test_places_name_first_and_deduplicates_explicit_fields() -> None:
    plan = make_plan(**{"return": ["founder", "name", "founder"]})

    assert compile_plan(plan, CATALOG) == ("companies.location=Spain.return(name, founder)")


def test_rejects_unknown_return_field() -> None:
    plan = make_plan(**{"return": ["founder_email"]})

    with pytest.raises(PlanCompilationError, match="unknown return field"):
        compile_plan(plan, CATALOG)


def test_rejects_unknown_order_by() -> None:
    plan = make_plan(order_by="name:asc", limit=5, limit_mention="5")

    with pytest.raises(PlanCompilationError, match="unknown order_by"):
        compile_plan(plan, CATALOG)


def test_rejects_unknown_filter() -> None:
    plan = make_plan(filters=[{"kind": "founder_eq", "mention": "Ada", "value": "Ada"}])

    with pytest.raises(PlanCompilationError, match="unknown filter kind"):
        compile_plan(plan, CATALOG)


@pytest.mark.parametrize("value", ["", "   ", " Spain", "Spain "])
def test_rejects_empty_or_padded_filter_value(value: str) -> None:
    plan = make_plan(filters=[{"kind": "location_eq", "mention": "Spain", "value": value}])

    with pytest.raises(PlanCompilationError):
        compile_plan(plan, CATALOG)


@pytest.mark.parametrize(
    "value",
    [
        "Spain.limit=100",
        "Google,Meta",
        "Spain.return(name)",
        "Spain\nlimit=100",
    ],
)
def test_rejects_filter_values_that_can_change_query_grammar(value: str) -> None:
    plan = make_plan(filters=[{"kind": "location_eq", "mention": "Spain", "value": value}])

    with pytest.raises(PlanCompilationError):
        compile_plan(plan, CATALOG)


def test_preserves_clause_order() -> None:
    plan = make_plan(
        root="startups",
        order_by="funding:desc",
        limit=5,
        limit_mention="5",
        **{"return": ["name", "funding"]},
    )

    assert compile_plan(plan, CATALOG) == (
        "startups.location=Spain.order_by=funding DESC.limit=5.return(name, funding)"
    )


@pytest.mark.parametrize("operation", ["retrieve_entity", "unsupported"])
def test_only_compiles_knowledge_queries(operation: str) -> None:
    plan = make_plan(operation=operation)

    with pytest.raises(PlanCompilationError):
        compile_plan(plan, CATALOG)


def test_rejects_unknown_root() -> None:
    with pytest.raises(PlanCompilationError, match="unknown root"):
        compile_plan(make_plan(root="investors"), CATALOG)


@pytest.mark.parametrize("limit", [0, 101, True])
def test_rejects_invalid_limit_if_model_validation_was_bypassed(limit: object) -> None:
    plan = make_plan(limit=1, limit_mention="1").model_copy(update={"limit": limit})

    with pytest.raises(PlanCompilationError, match="limit"):
        compile_plan(plan, CATALOG)


@pytest.mark.parametrize(
    ("field", "value"),
    [("entity", {"mention": "OpenAI"}), ("reason", "unsupported_property")],
)
def test_rejects_fields_from_other_operations(field: str, value: object) -> None:
    plan = make_plan(**{field: value})

    with pytest.raises(PlanCompilationError, match="knowledge query cannot include"):
        compile_plan(plan, CATALOG)
