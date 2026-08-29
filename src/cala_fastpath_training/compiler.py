from __future__ import annotations

from .catalog import Catalog
from .models import Plan


class PlanCompilationError(ValueError):
    pass


def compile_plan(plan: Plan, catalog: Catalog) -> str:
    if plan.operation != "knowledge_query":
        raise PlanCompilationError(f"operation {plan.operation!r} does not compile to Cala QL")
    if plan.root not in catalog.roots:
        raise PlanCompilationError(f"unknown root {plan.root!r}")
    clauses = [plan.root]
    for filter_value in plan.filters:
        spec = catalog.filters.get(filter_value.kind)
        if spec is None:
            raise PlanCompilationError(f"unknown filter kind {filter_value.kind!r}")
        value = str(filter_value.value).strip()
        if not value:
            raise PlanCompilationError(f"filter {filter_value.kind!r} has no normalized value")
        clauses.append(f"{'.'.join(spec.path)}{spec.operator}{value}")
    if plan.order_by:
        field, direction = plan.order_by.rsplit(":", 1)
        clauses.append(f"order_by={field} {direction.upper()}")
    if plan.limit is not None:
        clauses.append(f"limit={plan.limit}")
    if plan.return_fields:
        clauses.append(f"return({', '.join(plan.return_fields)})")
    return ".".join(clauses)
