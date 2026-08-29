from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from .models import FilterValue, GenerationRecord, Plan, TrainingExample

COMPONENTS = (
    "operation",
    "root",
    "filters",
    "return",
    "entity",
    "order_by",
    "limit",
    "reason",
)


class EvaluationInputError(ValueError):
    pass


def _text_value(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip().casefold()


def _filter_value(value: str | int | float) -> tuple[str, str]:
    if isinstance(value, str):
        return "text", _text_value(value)
    return "number", str(Decimal(str(value)).normalize())


def _filter_key(item: FilterValue) -> tuple[str, str, str]:
    value_type, value = _filter_value(item.value)
    return item.kind, value_type, value


def _filter_counts(plan: Plan) -> Counter[tuple[str, str, str]]:
    return Counter(_filter_key(item) for item in plan.filters)


def _entity_value(plan: Plan) -> str | None:
    if plan.entity is None:
        return None
    return _text_value(plan.entity.mention)


def _compare_plans(gold: Plan, predicted: Plan | None) -> dict[str, bool]:
    if predicted is None:
        return dict.fromkeys(COMPONENTS, False)
    return {
        "operation": predicted.operation == gold.operation,
        "root": predicted.root == gold.root,
        "filters": _filter_counts(predicted) == _filter_counts(gold),
        "return": set(predicted.return_fields) == set(gold.return_fields),
        "entity": _entity_value(predicted) == _entity_value(gold),
        "order_by": predicted.order_by == gold.order_by,
        "limit": predicted.limit == gold.limit,
        "reason": predicted.reason == gold.reason,
    }


def _status(record: GenerationRecord) -> str:
    if record.error is not None:
        return "error"
    if record.decision == "accepted":
        return "accepted"
    if record.decision == "abstained":
        return "rejected"
    if record.plan is None or record.plan.operation == "unsupported":
        return "rejected"
    return "accepted"


def _invalid_reasons(record: GenerationRecord) -> list[str]:
    reasons: list[str] = []
    if record.cala_query is not None and record.plan is None:
        reasons.append("query_without_plan")
    if record.decision == "accepted":
        if record.error is not None:
            reasons.append("accepted_with_error")
        if record.plan is None:
            reasons.append("accepted_without_plan")
        elif record.plan.operation == "unsupported":
            reasons.append("accepted_unsupported_plan")
    if record.decision == "abstained" and record.cala_query is not None:
        reasons.append("abstained_with_query")
    return reasons


def _filter_difference(
    left: Plan,
    right: Plan,
) -> Counter[tuple[str, str, str]]:
    return _filter_counts(left) - _filter_counts(right)


def _serialized_filters(
    counts: Counter[tuple[str, str, str]],
) -> list[dict[str, str | int]]:
    return [
        {"kind": kind, "value_type": value_type, "value": value, "count": count}
        for (kind, value_type, value), count in sorted(counts.items())
    ]


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _wilson_lower_bound(
    successes: int,
    trials: int,
    z: float = 1.6448536269514722,
) -> float:
    proportion = successes / trials
    z_squared = z * z
    denominator = 1 + z_squared / trials
    centre = proportion + z_squared / (2 * trials)
    margin = z * math.sqrt((proportion * (1 - proportion) + z_squared / (4 * trials)) / trials)
    return max(0.0, (centre - margin) / denominator)


def _index_unique(items: Iterable[Any], *, key: str, label: str) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for item in items:
        item_id = getattr(item, key)
        if item_id in indexed:
            raise EvaluationInputError(f"duplicate {label} id {item_id!r}")
        indexed[item_id] = item
    return indexed


def evaluate_generation_records(
    records: Iterable[GenerationRecord],
    gold_examples: Iterable[TrainingExample],
) -> dict[str, Any]:
    """Compare generation records with gold plans matched by case id."""
    predicted_by_id = _index_unique(records, key="case_id", label="generation record")
    gold_by_id = _index_unique(gold_examples, key="id", label="gold example")

    component_correct = Counter({component: 0 for component in COMPONENTS})
    cases: list[dict[str, Any]] = []
    accepted = 0
    rejected = 0
    accepted_exact = 0
    safe_rejections = 0
    extra_return_fields = 0
    extra_return_field_cases = 0
    lost_filters = 0
    lost_filter_cases = 0
    invalid_records = 0
    execution_errors = 0

    for case_id in sorted(gold_by_id):
        gold = gold_by_id[case_id]
        record = predicted_by_id.get(case_id)
        if record is None:
            comparison = dict.fromkeys(COMPONENTS, False)
            cases.append(
                {
                    "case_id": case_id,
                    "status": "missing",
                    "exact_plan": False,
                    **comparison,
                    "extra_return_fields": [],
                    "lost_filters": [],
                    "invalid_reasons": [],
                }
            )
            continue

        status = _status(record)
        comparison = _compare_plans(gold.plan, record.plan)
        exact_plan = all(comparison.values())
        component_correct.update(name for name, correct in comparison.items() if correct)
        invalid_reasons = _invalid_reasons(record)
        if invalid_reasons:
            invalid_records += 1
        if status == "accepted":
            accepted += 1
        elif status == "rejected":
            rejected += 1
            if not invalid_reasons and record.cala_query is None:
                safe_rejections += 1
        else:
            execution_errors += 1

        if status == "accepted" and exact_plan and not invalid_reasons:
            accepted_exact += 1

        extra_fields: list[str] = []
        lost_filter_counts: Counter[tuple[str, str, str]] = Counter()
        if status == "accepted" and record.plan is not None:
            extra_fields = sorted(set(record.plan.return_fields) - set(gold.plan.return_fields))
            lost_filter_counts = _filter_difference(gold.plan, record.plan)
            extra_return_fields += len(extra_fields)
            lost_filters += sum(lost_filter_counts.values())
            extra_return_field_cases += bool(extra_fields)
            lost_filter_cases += bool(lost_filter_counts)

        cases.append(
            {
                "case_id": case_id,
                "status": status,
                "exact_plan": exact_plan,
                **comparison,
                "extra_return_fields": extra_fields,
                "lost_filters": _serialized_filters(lost_filter_counts),
                "invalid_reasons": invalid_reasons,
            }
        )

    total = len(gold_by_id)
    exact_plan = sum(case["exact_plan"] for case in cases)
    missing_predictions = sorted(set(gold_by_id) - set(predicted_by_id))
    unexpected_predictions = sorted(set(predicted_by_id) - set(gold_by_id))
    result: dict[str, Any] = {
        "schema_version": "cala_fastpath_evaluation.v1",
        "total": total,
        "exact_plan": exact_plan,
        "exact_plan_rate": _ratio(exact_plan, total),
    }
    for component in COMPONENTS:
        correct = component_correct[component]
        result[component] = correct
        result[f"{component}_rate"] = _ratio(correct, total)
    result.update(
        {
            "accepted": accepted,
            "rejected": rejected,
            "accepted_exact": accepted_exact,
            "accepted_precision": _ratio(accepted_exact, accepted),
            "accepted_precision_lower_bound_95": (
                _wilson_lower_bound(accepted_exact, accepted) if accepted else None
            ),
            "coverage": _ratio(accepted, total),
            "unsafe_accepts": accepted - accepted_exact,
            "unsafe_accept_rate": _ratio(accepted - accepted_exact, accepted),
            "safe_rejections": safe_rejections,
            "extra_return_fields": extra_return_fields,
            "extra_return_field_cases": extra_return_field_cases,
            "lost_filters": lost_filters,
            "lost_filter_cases": lost_filter_cases,
            "invalid_records": invalid_records,
            "execution_errors": execution_errors,
            "missing_predictions": missing_predictions,
            "unexpected_predictions": unexpected_predictions,
            "cases": cases,
        }
    )
    return result


def evaluate_generation_records_by_system(
    records: Iterable[GenerationRecord],
    gold_examples: Iterable[TrainingExample],
) -> dict[str, Any]:
    """Evaluate each generation system against the same gold examples."""
    gold = list(gold_examples)
    _index_unique(gold, key="id", label="gold example")
    records_by_system: dict[str, list[GenerationRecord]] = {}
    for record in records:
        records_by_system.setdefault(record.system, []).append(record)

    return {
        "schema_version": "cala_fastpath_evaluation_by_system.v1",
        "total": len(gold),
        "systems": {
            system: evaluate_generation_records(system_records, gold)
            for system, system_records in sorted(records_by_system.items())
        },
    }
