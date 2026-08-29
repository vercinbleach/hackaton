from __future__ import annotations

import math

from .catalog import Catalog
from .models import Plan


class PlanCompilationError(ValueError):
    pass


_FORBIDDEN_VALUE_CHARACTERS = frozenset(".,=!<>()")


def _serialize_filter_value(value: str | int | float, *, kind: str) -> str:
    if isinstance(value, bool):
        raise PlanCompilationError(f"filter {kind!r} has an invalid boolean value")
    if isinstance(value, float) and not math.isfinite(value):
        raise PlanCompilationError(f"filter {kind!r} has a non-finite value")

    serialized = str(value)
    if not serialized or serialized != serialized.strip():
        raise PlanCompilationError(f"filter {kind!r} has an empty or padded value")
    if any(character.isspace() and character != " " for character in serialized):
        raise PlanCompilationError(f"filter {kind!r} contains control whitespace")
    if any(character in _FORBIDDEN_VALUE_CHARACTERS for character in serialized):
        raise PlanCompilationError(f"filter {kind!r} contains Cala QL grammar characters")
    return serialized


def _closed_projection(plan: Plan, catalog: Catalog) -> list[str]:
    unknown_returns = set(plan.return_fields) - set(catalog.returns)
    if unknown_returns:
        raise PlanCompilationError(f"unknown return fields: {', '.join(sorted(unknown_returns))}")

    projection = ["name"]
    for field in plan.return_fields:
        if field not in projection:
            projection.append(field)
    return projection


def validate_plan_for_fastpath(plan: Plan, catalog: Catalog) -> None:
    if plan.operation != "knowledge_query":
        raise PlanCompilationError(f"operation {plan.operation!r} is not a FastPath query")
    if plan.root not in catalog.roots:
        raise PlanCompilationError(f"unknown root {plan.root!r}")
    if plan.entity is not None:
        raise PlanCompilationError("knowledge query cannot include an entity")
    if plan.reason is not None:
        raise PlanCompilationError("knowledge query cannot include an unsupported reason")
    _closed_projection(plan, catalog)
    if plan.order_by is not None and plan.order_by not in catalog.order_by:
        raise PlanCompilationError(f"unknown order_by {plan.order_by!r}")
    if plan.order_by is not None and plan.limit is None:
        raise PlanCompilationError("ordered FastPath query requires an explicit limit")
    if plan.limit is not None and (
        isinstance(plan.limit, bool)
        or not isinstance(plan.limit, int)
        or not 1 <= plan.limit <= 100
    ):
        raise PlanCompilationError("limit must be an integer between 1 and 100")
    if not plan.filters and not (plan.order_by and plan.limit):
        raise PlanCompilationError("unconstrained collection query is unsafe for FastPath")

    seen_filters: set[str] = set()
    for filter_value in plan.filters:
        if filter_value.kind in seen_filters:
            raise PlanCompilationError(f"duplicate filter kind {filter_value.kind!r}")
        seen_filters.add(filter_value.kind)
        spec = catalog.filters.get(filter_value.kind)
        if spec is None:
            raise PlanCompilationError(f"unknown filter kind {filter_value.kind!r}")
        _serialize_filter_value(filter_value.value, kind=filter_value.kind)


def compile_plan(plan: Plan, catalog: Catalog) -> str:
    validate_plan_for_fastpath(plan, catalog)
    clauses = [plan.root]
    for filter_value in plan.filters:
        spec = catalog.filters[filter_value.kind]
        value = _serialize_filter_value(filter_value.value, kind=filter_value.kind)
        clauses.append(f"{'.'.join(spec.path)}{spec.operator}{value}")
    if plan.order_by:
        field, direction = plan.order_by.rsplit(":", 1)
        clauses.append(f"order_by={field} {direction.upper()}")
    if plan.limit is not None:
        clauses.append(f"limit={plan.limit}")
    clauses.append(f"return({', '.join(_closed_projection(plan, catalog))})")
    return ".".join(clauses)
