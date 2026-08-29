from __future__ import annotations

import json

from cala_fastpath_training.evaluation import (
    evaluate_generation_records,
    evaluate_generation_records_by_system,
)
from cala_fastpath_training.models import GenerationRecord, Plan, TrainingExample


def _gold(case_id: str, plan: Plan, text: str) -> TrainingExample:
    return TrainingExample(
        id=case_id,
        group=f"group-{case_id}",
        language="es",
        text=text,
        plan=plan,
    )


def _record(
    case_id: str,
    plan: Plan | None,
    *,
    decision: str | None = "accepted",
    error: str | None = None,
    system: str = "test",
) -> GenerationRecord:
    return GenerationRecord.model_validate(
        {
            "case_id": case_id,
            "query": "query",
            "system": system,
            "model": "test",
            "plan": plan,
            "cala_query": "companies.return(name)" if decision == "accepted" and plan else None,
            "latency_ms": 1,
            "decision": decision,
            "error": error,
        }
    )


def _company_plan(*, returns: list[str] | None = None, with_filter: bool = True) -> Plan:
    return Plan.model_validate(
        {
            "operation": "knowledge_query",
            "root": "companies",
            "filters": (
                [{"kind": "previous_job_eq", "mention": "Google", "value": "Google"}]
                if with_filter
                else []
            ),
            "return": returns or ["name"],
        }
    )


def test_exact_plan_uses_return_set_and_conservative_filter_normalization() -> None:
    gold_plan = _company_plan(returns=["name", "founder"])
    predicted_plan = Plan.model_validate(
        {
            "operation": "knowledge_query",
            "root": "companies",
            "filters": [{"kind": "previous_job_eq", "mention": "google", "value": " google "}],
            "return": ["founder", "name"],
        }
    )

    result = evaluate_generation_records(
        [_record("exact", predicted_plan)],
        [_gold("exact", gold_plan, "Empresas fundadas por exempleados de Google")],
    )

    assert result["exact_plan"] == 1
    assert result["filters"] == 1
    assert result["return"] == 1
    assert result["accepted_precision"] == 1.0


def test_extra_return_field_breaks_exactness_and_is_counted() -> None:
    result = evaluate_generation_records(
        [_record("extra", _company_plan(returns=["name", "founder"]))],
        [
            _gold(
                "extra",
                _company_plan(returns=["name"]),
                "Empresas fundadas por exempleados de Google",
            )
        ],
    )

    assert result["exact_plan"] == 0
    assert result["return"] == 0
    assert result["extra_return_fields"] == 1
    assert result["cases"][0]["extra_return_fields"] == ["founder"]
    assert result["accepted_precision"] == 0.0


def test_lost_filter_preserves_multiplicity() -> None:
    gold_plan = Plan.model_validate(
        {
            "operation": "knowledge_query",
            "root": "companies",
            "filters": [
                {"kind": "industry_eq", "mention": "AI", "value": "AI"},
                {"kind": "industry_eq", "mention": "AI", "value": "AI"},
            ],
            "return": ["name"],
        }
    )
    predicted_plan = Plan.model_validate(
        {
            "operation": "knowledge_query",
            "root": "companies",
            "filters": [{"kind": "industry_eq", "mention": "AI", "value": "AI"}],
            "return": ["name"],
        }
    )

    result = evaluate_generation_records(
        [_record("lost", predicted_plan)],
        [_gold("lost", gold_plan, "Empresas de AI")],
    )

    assert result["filters"] == 0
    assert result["lost_filters"] == 1
    assert result["cases"][0]["lost_filters"] == [
        {"kind": "industry_eq", "value_type": "text", "value": "ai", "count": 1}
    ]


def test_safe_rejection_does_not_look_like_an_error_or_lost_filter() -> None:
    result = evaluate_generation_records(
        [_record("reject", None, decision="abstained")],
        [
            _gold(
                "reject",
                _company_plan(),
                "Empresas fundadas por exempleados de Google",
            )
        ],
    )

    assert result["accepted"] == 0
    assert result["rejected"] == 1
    assert result["safe_rejections"] == 1
    assert result["coverage"] == 0.0
    assert result["lost_filters"] == 0
    assert result["execution_errors"] == 0
    assert result["cases"][0]["status"] == "rejected"


def test_execution_error_is_not_counted_as_a_rejection() -> None:
    result = evaluate_generation_records(
        [_record("error", None, decision=None, error="model failed")],
        [
            _gold(
                "error",
                _company_plan(),
                "Empresas fundadas por exempleados de Google",
            )
        ],
    )

    assert result["rejected"] == 0
    assert result["execution_errors"] == 1
    assert result["cases"][0]["status"] == "error"


def test_result_is_json_serializable_and_stable_across_input_order() -> None:
    gold = [
        _gold("a", _company_plan(), "Empresas fundadas por exempleados de Google"),
        _gold("b", _company_plan(), "Empresas fundadas por exempleados de Google"),
    ]
    records = [
        _record("b", _company_plan()),
        _record("a", _company_plan()),
    ]

    first = evaluate_generation_records(records, gold)
    second = evaluate_generation_records(reversed(records), reversed(gold))

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True))["total"] == 2


def test_accepted_record_without_plan_is_reported_as_invalid() -> None:
    result = evaluate_generation_records(
        [_record("invalid", None, decision="accepted")],
        [
            _gold(
                "invalid",
                _company_plan(),
                "Empresas fundadas por exempleados de Google",
            )
        ],
    )

    assert result["accepted"] == 1
    assert result["accepted_exact"] == 0
    assert result["invalid_records"] == 1
    assert result["cases"][0]["invalid_reasons"] == ["accepted_without_plan"]


def test_five_exact_accepts_have_a_weak_wilson_lower_bound() -> None:
    gold = [
        _gold(
            f"case-{index}",
            _company_plan(),
            "Empresas fundadas por exempleados de Google",
        )
        for index in range(5)
    ]
    records = [_record(row.id, row.plan) for row in gold]

    result = evaluate_generation_records(records, gold)

    assert result["accepted_precision"] == 1.0
    assert result["accepted_precision_lower_bound_95"] < 0.7
    assert result["unsafe_accepts"] == 0
    assert result["unsafe_accept_rate"] == 0.0


def test_groups_records_by_system_against_the_same_gold() -> None:
    gold = [
        _gold(
            "shared-case",
            _company_plan(),
            "Empresas fundadas por exempleados de Google",
        )
    ]
    records = [
        _record("shared-case", _company_plan(), system="exact-system"),
        _record(
            "shared-case",
            _company_plan(returns=["name", "founder"]),
            system="unsafe-system",
        ),
    ]

    result = evaluate_generation_records_by_system(records, gold)

    assert list(result["systems"]) == ["exact-system", "unsafe-system"]
    assert result["systems"]["exact-system"]["accepted_exact"] == 1
    assert result["systems"]["unsafe-system"]["unsafe_accepts"] == 1
    assert result["systems"]["unsafe-system"]["unsafe_accept_rate"] == 1.0
