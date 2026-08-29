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


def _compiled(plan: Plan, catalog: Catalog) -> str | None:
    try:
        return compile_plan(plan, catalog)
    except PlanCompilationError:
        return None


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
            return GenerationRecord(
                case_id=case.id,
                query=case.query,
                system=self.system,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                plan=plan,
                cala_query=_compiled(plan, self.catalog),
                latency_ms=(time.perf_counter() - started) * 1000,
                usage=usage,
                raw=compact_openai_raw(raw),
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
                "plan_tags": {
                    "labels": self.catalog.classification_labels,
                    "multi_label": True,
                }
            }
            structures = {
                name: [
                    {
                        "name": "value",
                        "dtype": "str",
                        "description": spec.description,
                    }
                ]
                for name, spec in self.catalog.filters.items()
            }
            structures["entity_name"] = [
                {
                    "name": "value",
                    "dtype": "str",
                    "description": "Exact name of the entity requested by the user",
                }
            ]
            structures["limit_value"] = [
                {
                    "name": "value",
                    "dtype": "str",
                    "description": "Maximum result count stated by the user",
                }
            ]
            tags = extractor.classify_text(
                case.query,
                tasks,
                threshold=self.threshold,
                include_confidence=True,
            )
            extracted = extractor.extract_json(
                case.query,
                structures,
                threshold=self.threshold,
                include_confidence=True,
            )
            plan = self._to_plan(tags, extracted)
            return GenerationRecord(
                case_id=case.id,
                query=case.query,
                system=self.system,
                model=self.model,
                plan=plan,
                cala_query=_compiled(plan, self.catalog),
                latency_ms=(time.perf_counter() - started) * 1000,
                raw={"classifications": tags, "structures": extracted},
            )
        except Exception as exc:
            return GenerationRecord(
                case_id=case.id,
                query=case.query,
                system=self.system,
                model=self.model,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )

    def _to_plan(self, tags: dict[str, Any], extracted: dict[str, Any]) -> Plan:
        labels = [
            item.get("label", "") for item in tags.get("plan_tags", []) if isinstance(item, dict)
        ]

        def selected(prefix: str) -> list[str]:
            return [label.removeprefix(prefix) for label in labels if label.startswith(prefix)]

        filters: list[dict[str, Any]] = []
        for kind in self.catalog.filters:
            for record in extracted.get(kind, []):
                value = self._structure_text(record)
                if value:
                    filters.append({"kind": kind, "mention": value, "value": value})

        operation = next(iter(selected("operation:")), None)
        roots = selected("root:")
        returns = selected("return:")
        reasons = selected("reason:")
        entity_text = self._first_structure_text(extracted.get("entity_name", []))
        limit_text = self._first_structure_text(extracted.get("limit_value", []))

        if operation is None:
            if reasons:
                operation = "unsupported"
            elif entity_text and not filters:
                operation = "retrieve_entity"
            else:
                operation = "knowledge_query"

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
                "order_by": next(iter(selected("order_by:")), None),
                "limit": limit,
                "limit_mention": limit_text if limit is not None else None,
                "reason": reasons[0] if reasons else None,
            }
        )

    @staticmethod
    def _structure_text(record: Any) -> str | None:
        if not isinstance(record, dict):
            return None
        value = record.get("value")
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("text"), str):
            return value["text"]
        return None

    @classmethod
    def _first_structure_text(cls, records: Any) -> str | None:
        if not isinstance(records, list):
            return None
        for record in records:
            if value := cls._structure_text(record):
                return value
        return None


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
