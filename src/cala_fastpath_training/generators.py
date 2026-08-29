from __future__ import annotations

import os
import stat
import time
from pathlib import Path
from typing import Any, Protocol

from .catalog import Catalog
from .compiler import PlanCompilationError, compile_plan
from .models import BenchmarkCase, GenerationRecord, Plan
from .openai_responses import OpenAIResponsesClient, compact_openai_raw

BASE_INSTRUCTIONS = """Convert the user's natural-language request into a Cala query plan.
Return only the structured plan. Include only fields explicitly requested or required to identify
the result."""


class Generator(Protocol):
    system: str
    model: str

    def generate(self, case: BenchmarkCase) -> GenerationRecord: ...


class PlanExtractionError(ValueError):
    pass


def _compile_decision(plan: Plan, catalog: Catalog) -> tuple[str | None, str, str | None]:
    try:
        return compile_plan(plan, catalog), "accepted", None
    except PlanCompilationError as exc:
        return None, "abstained", str(exc)


class OpenAIGenerator:
    def __init__(
        self,
        *,
        system: str,
        model: str,
        reasoning_effort: str,
        instructions: str,
        catalog: Catalog,
    ) -> None:
        self.system = system
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.instructions = instructions
        self.catalog = catalog

    def generate(self, case: BenchmarkCase) -> GenerationRecord:
        started = time.perf_counter()
        try:
            with OpenAIResponsesClient.from_environment() as client:
                plan, raw, usage = client.generate_plan(
                    case.query,
                    model=self.model,
                    reasoning_effort=self.reasoning_effort,
                    instructions=self.instructions,
                    catalog=self.catalog,
                )
            cala_query, decision, abstention_reason = _compile_decision(plan, self.catalog)
            return GenerationRecord(
                case_id=case.id,
                query=case.query,
                system=self.system,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                plan=plan,
                cala_query=cala_query,
                latency_ms=(time.perf_counter() - started) * 1000,
                usage=usage,
                raw=compact_openai_raw(raw),
                decision=decision,
                abstention_reason=abstention_reason,
            )
        except Exception as exc:
            return GenerationRecord(
                case_id=case.id,
                query=case.query,
                system=self.system,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )


class GLiNERBaseGenerator:
    system = "base"

    def __init__(self, *, model: str, catalog: Catalog, threshold: float = 0.5) -> None:
        self.model = model
        self.catalog = catalog
        self.threshold = threshold
        self._extractor: Any = None

    def _load(self) -> Any:
        if self._extractor is None:
            from gliner2 import GLiNER2

            self._extractor = GLiNER2.from_pretrained(self.model)
        return self._extractor

    def generate(self, case: BenchmarkCase) -> GenerationRecord:
        started = time.perf_counter()
        try:
            extractor = self._load()
            tasks = {
                task: {**config, "cls_threshold": self.threshold}
                for task, config in self.catalog.classification_tasks.items()
            }
            tags = extractor.classify_text(
                case.query,
                tasks,
                threshold=self.threshold,
                include_confidence=True,
            )
            extracted = extractor.extract_entities(
                case.query,
                self.catalog.ner_entities,
                threshold=self.threshold,
                include_confidence=True,
                include_spans=True,
            )
            raw = {"classifications": tags, "entities": extracted}
            try:
                plan = self._to_plan(tags, extracted)
            except PlanExtractionError as exc:
                return GenerationRecord(
                    case_id=case.id,
                    query=case.query,
                    system=self.system,
                    model=self.model,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    raw=raw,
                    decision="abstained",
                    abstention_reason=str(exc),
                    threshold=self.threshold,
                )
            cala_query, decision, abstention_reason = _compile_decision(plan, self.catalog)
            return GenerationRecord(
                case_id=case.id,
                query=case.query,
                system=self.system,
                model=self.model,
                plan=plan,
                cala_query=cala_query,
                latency_ms=(time.perf_counter() - started) * 1000,
                raw=raw,
                decision=decision,
                abstention_reason=abstention_reason,
                threshold=self.threshold,
            )
        except Exception as exc:
            return GenerationRecord(
                case_id=case.id,
                query=case.query,
                system=self.system,
                model=self.model,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
                threshold=self.threshold,
            )

    def _to_plan(self, tags: dict[str, Any], extracted: dict[str, Any]) -> Plan:
        labels = self._selected_labels(tags, "plan_labels")

        def selected(prefix: str) -> list[str]:
            return [label.removeprefix(prefix) for label in labels if label.startswith(prefix)]

        operations = selected("operation:")
        roots = selected("root:")
        explicit_returns = selected("return:")
        filter_intents = selected("filter:")
        orders = selected("order_by:")
        reasons = selected("reason:")

        if len(operations) != 1:
            raise PlanExtractionError("operation must have exactly one confident prediction")
        operation = operations[0]
        if len(roots) > 1:
            raise PlanExtractionError("root has conflicting confident predictions")
        if len(reasons) > 1:
            raise PlanExtractionError("unsupported reason has conflicting predictions")
        if len(orders) > 1:
            raise PlanExtractionError("order_by has conflicting confident predictions")

        filter_values: dict[str, list[str]] = {
            kind: self._entity_texts(extracted, spec.structure)
            for kind, spec in self.catalog.filters.items()
        }
        predicted_filter_kinds = set(filter_intents)
        extracted_filter_kinds = {kind for kind, values in filter_values.items() if values}
        if predicted_filter_kinds != extracted_filter_kinds:
            missing = sorted(predicted_filter_kinds - extracted_filter_kinds)
            extra = sorted(extracted_filter_kinds - predicted_filter_kinds)
            raise PlanExtractionError(
                f"filter label/extraction mismatch: missing={missing}, extra={extra}"
            )

        filters: list[dict[str, Any]] = []
        for kind in filter_intents:
            if kind not in self.catalog.filters:
                raise PlanExtractionError(f"unknown filter intent {kind!r}")
            values = filter_values[kind]
            if len(values) != 1:
                field = self.catalog.filters[kind].field
                raise PlanExtractionError(f"filter intent {kind!r} requires one extracted {field}")
            filters.append({"kind": kind, "mention": values[0], "value": values[0]})

        entity_values = self._entity_texts(extracted, "target_entity")
        limit_values = self._entity_texts(extracted, "result_limit")
        if len(entity_values) > 1:
            raise PlanExtractionError("multiple entity names extracted")
        if len(limit_values) > 1:
            raise PlanExtractionError("multiple limits extracted")

        entity_text = entity_values[0] if entity_values else None
        limit_text = limit_values[0] if limit_values else None
        if limit_text is not None and not limit_text.isdigit():
            raise PlanExtractionError("limit is not an integer")

        if operation == "unsupported":
            if len(reasons) != 1:
                raise PlanExtractionError("unsupported operation requires one confident reason")
            if roots or explicit_returns or filters or entity_text or orders or limit_text:
                raise PlanExtractionError("unsupported operation contains executable plan fields")
        elif operation == "retrieve_entity":
            if len(roots) != 1 or entity_text is None or not explicit_returns:
                raise PlanExtractionError(
                    "retrieve_entity requires one root, one entity, and a result projection"
                )
            if filters or reasons or orders or limit_text:
                raise PlanExtractionError("retrieve_entity contains incompatible plan fields")
        elif operation == "knowledge_query":
            if len(roots) != 1:
                raise PlanExtractionError("knowledge_query requires one confident root")
            if entity_text or reasons:
                raise PlanExtractionError("knowledge_query contains incompatible plan fields")
            if not filters and not (orders and limit_text):
                raise PlanExtractionError("unconstrained collection query is unsafe for FastPath")
        else:
            raise PlanExtractionError(f"unknown operation {operation!r}")

        returns = ["name"] if operation == "knowledge_query" else []
        returns.extend(
            field for field in explicit_returns if field != "name" and field not in returns
        )
        limit = int(limit_text) if limit_text and limit_text.isdigit() else None
        return Plan.model_validate(
            {
                "operation": operation,
                "root": roots[0] if roots else None,
                "filters": filters,
                "return": returns,
                "entity": {"mention": entity_text}
                if operation == "retrieve_entity" and entity_text
                else None,
                "order_by": orders[0] if orders else None,
                "limit": limit,
                "limit_mention": limit_text if limit is not None else None,
                "reason": reasons[0] if reasons else None,
            }
        )

    @staticmethod
    def _entity_text(record: Any, threshold: float) -> str | None:
        if isinstance(record, str):
            return record
        if not isinstance(record, dict):
            return None
        value = record.get("text")
        confidence = record.get("confidence", 1.0)
        if (
            isinstance(value, str)
            and isinstance(confidence, int | float)
            and confidence >= threshold
        ):
            return value
        return None

    def _entity_texts(self, extracted: Any, label: str) -> list[str]:
        records: Any = extracted
        if isinstance(records, dict) and "entities" in records:
            records = records["entities"]
        if isinstance(records, dict):
            records = records.get(label, [])
        elif isinstance(records, list):
            records = [
                record
                for record in records
                if isinstance(record, dict) and record.get("label") == label
            ]
        if not isinstance(records, list):
            records = [records]
        return [
            value
            for record in records
            if (value := self._entity_text(record, self.threshold)) is not None
        ]

    def _selected_labels(self, tags: dict[str, Any], task: str) -> list[str]:
        raw = tags.get(task, [])
        records = raw if isinstance(raw, list) else [raw]
        labels: list[str] = []
        for item in records:
            if not isinstance(item, dict) or not isinstance(item.get("label"), str):
                continue
            confidence = item.get("confidence")
            if isinstance(confidence, int | float) and confidence >= self.threshold:
                labels.append(item["label"])
        return list(dict.fromkeys(labels))


def load_skill(path: Path, allowed_root: Path) -> str:
    """Securely load a skill file with comprehensive validation.

    This prevents file exfiltration by:
    - Requiring the path to stay within allowed_root
    - Rejecting symlinks and junctions in the path hierarchy
    - Rejecting hardlinks (st_nlink != 1)
    - Opening with O_NOFOLLOW where available
    - Verifying the file descriptor with fstat
    - Reading from the same descriptor to avoid TOCTOU

    Args:
        path: The skill file path to load
        allowed_root: The root directory that must contain the skill file

    Raises:
        ValueError: If validation fails or the file is unsafe
    """
    # Normalize paths
    allowed_root = Path(os.path.abspath(allowed_root)).resolve()
    unresolved = path if path.is_absolute() else Path(os.path.abspath(path))

    # Check for symlinks/junctions in the path hierarchy
    current = unresolved
    while current != current.parent:
        if current.exists() and (
            current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction())
        ):
            raise ValueError(f"skill path cannot use symbolic links or junctions: {path}")
        current = current.parent

    # Verify the file exists
    if not unresolved.exists():
        raise ValueError(f"skill file does not exist: {path}")

    # Verify it's a file
    if not unresolved.is_file():
        raise ValueError(f"skill path must be a file: {path}")

    # Resolve and check containment
    resolved = unresolved.resolve()
    if not resolved.is_relative_to(allowed_root):
        raise ValueError(f"skill path must stay within {allowed_root}: {path}")

    # Open with O_NOFOLLOW where available to prevent symlink following
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        fd = os.open(resolved, flags)
    except OSError as exc:
        raise ValueError(f"cannot open skill file: {path}") from exc

    try:
        # Verify the file descriptor with fstat
        st = os.fstat(fd)

        # Reject if not a regular file
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"skill path must be a regular file: {path}")

        # Reject hardlinks (st_nlink != 1)
        if st.st_nlink != 1:
            raise ValueError(f"skill file cannot have multiple hard links: {path}")

        # Read to EOF from the same descriptor to avoid TOCTOU and short reads.
        handle = os.fdopen(fd, encoding="utf-8")
        fd = -1
        with handle:
            content = handle.read()
    finally:
        if fd >= 0:
            os.close(fd)

    # Strip frontmatter if present
    if content.startswith("---\n"):
        _, _, remainder = content.partition("\n---\n")
        return remainder.strip()
    return content.strip()


def build_generators(
    systems: list[str],
    *,
    catalog: Catalog,
    skill_path: Path,
    skills_root: Path,
    openai_model: str | None = None,
    reasoning_effort: str | None = None,
    base_model: str | None = None,
    threshold: float = 0.5,
) -> list[Generator]:
    allowed = {"openai", "base", "openai-skill"}
    unknown = set(systems) - allowed
    if unknown:
        raise ValueError(f"unknown systems: {', '.join(sorted(unknown))}")
    openai_model = openai_model or os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
    reasoning_effort = reasoning_effort or os.environ.get("OPENAI_REASONING_EFFORT", "high")
    allowed_efforts = {"none", "low", "medium", "high", "xhigh", "max"}
    if reasoning_effort not in allowed_efforts:
        raise ValueError(f"unknown OpenAI reasoning effort: {reasoning_effort}")
    base_model = base_model or os.environ.get("GLINER_BASE_MODEL", "fastino/gliner2-multi-v1")
    generators: list[Generator] = []
    for system in systems:
        if system == "openai":
            generators.append(
                OpenAIGenerator(
                    system=system,
                    model=openai_model,
                    reasoning_effort=reasoning_effort,
                    instructions=BASE_INSTRUCTIONS,
                    catalog=catalog,
                )
            )
        elif system == "openai-skill":
            generators.append(
                OpenAIGenerator(
                    system=system,
                    model=openai_model,
                    reasoning_effort=reasoning_effort,
                    instructions=f"{BASE_INSTRUCTIONS}\n\n{load_skill(skill_path, skills_root)}",
                    catalog=catalog,
                )
            )
        else:
            generators.append(
                GLiNERBaseGenerator(model=base_model, catalog=catalog, threshold=threshold)
            )
    return generators
