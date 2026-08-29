from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv

from .catalog import load_catalog
from .dataset import (
    dataset_summary,
    grouped_split,
    read_jsonl,
    to_pioneer_row,
    validate_examples,
)
from .generators import build_generators
from .models import BenchmarkCase, JsonObject, PioneerRow, TrainingExample
from .pioneer import PioneerClient
from .seed_data import bootstrap_examples

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "training" / "config" / "catalog.json"
DEFAULT_QUESTIONS = ROOT / "benchmark" / "data" / "questions.jsonl"
DEFAULT_SKILL = ROOT / "benchmark" / "skills" / "cala-query" / "SKILL.md"
load_dotenv(ROOT / ".env")
app = typer.Typer(no_args_is_help=True, help="Cala FastPath GLiNER2 training pipeline")


def _resolve_output_path(path: Path, allowed_subdir: Path, option_name: str) -> Path:
    project_root = Path(os.path.abspath(ROOT))
    allowed_root = (project_root / allowed_subdir).resolve()
    unresolved = path if path.is_absolute() else project_root / path
    unresolved = Path(os.path.abspath(unresolved))
    current = unresolved
    while current != project_root and current.is_relative_to(project_root):
        if current.exists() and (
            current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction())
        ):
            raise typer.BadParameter(f"{option_name} cannot use symbolic links or junctions")
        current = current.parent
    resolved = unresolved.resolve()
    if not resolved.is_relative_to(allowed_root):
        raise typer.BadParameter(f"{option_name} must stay within {allowed_root}")
    return resolved


def _write_output_text(path: Path, content: str, allowed_subdir: Path, option_name: str) -> Path:
    resolved = _resolve_output_path(path, allowed_subdir, option_name)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_output_path(resolved, allowed_subdir, option_name)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=resolved.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        resolved = _resolve_output_path(resolved, allowed_subdir, option_name)
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved


def _write_output_jsonl(
    path: Path,
    rows: Iterable[TrainingExample | PioneerRow],
    allowed_subdir: Path,
    option_name: str,
) -> Path:
    content = "".join(
        json.dumps(
            row.model_dump(by_alias=True, exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for row in rows
    )
    return _write_output_text(path, content, allowed_subdir, option_name)


def _json(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True, exclude_none=True)
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _required_id(value: JsonObject, label: str) -> str:
    identifier = value.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise typer.BadParameter(f"Pioneer did not return a {label} ID")
    return identifier


def _read_benchmark_cases(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if "query" not in value and "text" in value:
            value = {"id": value["id"], "query": value["text"]}
        try:
            cases.append(BenchmarkCase.model_validate(value))
        except ValueError as exc:
            raise typer.BadParameter(f"invalid benchmark row {line_number}: {exc}") from exc
    if not cases:
        raise typer.BadParameter(f"no benchmark cases found in {path}")
    return cases


@app.command()
def generate(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)] = DEFAULT_QUESTIONS,
    systems: Annotated[str, typer.Option()] = "openai,base,openai-skill",
    output: Annotated[Path, typer.Option()] = Path("benchmark/runs/latest.jsonl"),
    catalog: Annotated[Path, typer.Option()] = DEFAULT_CATALOG,
    skill: Annotated[Path, typer.Option()] = DEFAULT_SKILL,
    openai_model: Annotated[str | None, typer.Option()] = None,
    reasoning_effort: Annotated[str | None, typer.Option()] = None,
    base_model: Annotated[str | None, typer.Option()] = None,
    threshold: Annotated[float, typer.Option(min=0, max=1)] = 0.5,
) -> None:
    """Generate comparable plans without grading them."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    output = _resolve_output_path(output, Path("benchmark/runs"), "--output")
    selected = [item.strip() for item in systems.split(",") if item.strip()]
    try:
        generators = build_generators(
            selected,
            catalog=load_catalog(catalog),
            skill_path=skill,
            openai_model=openai_model,
            reasoning_effort=reasoning_effort,
            base_model=base_model,
            threshold=threshold,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    records = []
    for case in _read_benchmark_cases(input_path):
        for generator in generators:
            typer.echo(f"[{case.id}] {generator.system} ({generator.model})", err=True)
            records.append(generator.generate(case))

    output = _write_output_text(
        output,
        "".join(
            json.dumps(
                record.model_dump(by_alias=True, exclude_none=True),
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        Path("benchmark/runs"),
        "--output",
    )
    _json(
        {
            "output": str(output),
            "records": len(records),
            "succeeded": sum(record.error is None for record in records),
            "failed": sum(record.error is not None for record in records),
        }
    )


@app.command()
def bootstrap(
    output: Annotated[Path, typer.Option()] = Path("training/data/examples.jsonl"),
    catalog: Annotated[Path, typer.Option()] = DEFAULT_CATALOG,
) -> None:
    """Create the initial canonical examples."""
    output = _resolve_output_path(output, Path("training/data"), "--output")
    rows = bootstrap_examples()
    validate_examples(rows, load_catalog(catalog))
    output = _write_output_jsonl(output, rows, Path("training/data"), "--output")
    _json(dataset_summary(rows))


@app.command("validate")
def validate_command(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    catalog: Annotated[Path, typer.Option()] = DEFAULT_CATALOG,
) -> None:
    """Validate canonical JSONL."""
    rows = read_jsonl(input_path)
    validate_examples(rows, load_catalog(catalog))
    _json(dataset_summary(rows))


@app.command("build")
def build_command(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()] = Path("training/artifacts/v0"),
    seed: Annotated[int, typer.Option()] = 42,
    train_ratio: Annotated[float, typer.Option()] = 0.7,
    validation_ratio: Annotated[float, typer.Option()] = 0.15,
    catalog: Annotated[Path, typer.Option()] = DEFAULT_CATALOG,
) -> None:
    """Create grouped splits and Pioneer files."""
    output_dir = _resolve_output_path(output_dir, Path("training/artifacts"), "--output-dir")
    loaded_catalog = load_catalog(catalog)
    rows = read_jsonl(input_path)
    validate_examples(rows, loaded_catalog)
    splits = grouped_split(
        rows,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    manifest: dict[str, Any] = {
        "source": str(input_path),
        "seed": seed,
        "schema": loaded_catalog.inference_schema(),
        "splits": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = _resolve_output_path(output_dir, Path("training/artifacts"), "--output-dir")
    for name in ("train", "validation", "test"):
        split_rows = getattr(splits, name)
        _write_output_jsonl(
            output_dir / "canonical" / f"{name}.jsonl",
            split_rows,
            Path("training/artifacts"),
            "--output-dir",
        )
        _write_output_jsonl(
            output_dir / "pioneer" / f"{name}.jsonl",
            (to_pioneer_row(row) for row in split_rows),
            Path("training/artifacts"),
            "--output-dir",
        )
        manifest["splits"][name] = dataset_summary(split_rows)
    _write_output_text(
        output_dir / "schema.json",
        json.dumps(
            loaded_catalog.inference_schema(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        Path("training/artifacts"),
        "--output-dir",
    )
    _write_output_text(
        output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        Path("training/artifacts"),
        "--output-dir",
    )
    _json(manifest["splits"])


@app.command()
def models() -> None:
    """List live trainable Pioneer encoders."""
    with PioneerClient.from_environment() as client:
        _json(client.list_trainable_models())


@app.command()
def upload(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    name: Annotated[str, typer.Option()],
    purpose: Annotated[str, typer.Option(help="training or evaluation")],
    wait: Annotated[bool, typer.Option()] = False,
) -> None:
    """Upload a JSONL dataset to Pioneer."""
    if purpose not in {"training", "evaluation"}:
        raise typer.BadParameter("purpose must be training or evaluation")
    with PioneerClient.from_environment() as client:
        uploaded = client.upload_dataset(input_path, dataset_name=name, purpose=purpose)
        result = uploaded.model_dump()
        if wait:
            result["ready"] = client.wait_for_dataset(uploaded.dataset_name)
    _json(result)


@app.command()
def train(
    dataset: Annotated[str, typer.Option()],
    model_name: Annotated[str, typer.Option()] = "cala-fastpath-v0",
    base_model: Annotated[str, typer.Option()] = "fastino/gliner2-multi-v1",
    epochs: Annotated[int, typer.Option(min=1)] = 5,
    learning_rate: Annotated[float, typer.Option(min=0)] = 5e-5,
    wait: Annotated[bool, typer.Option()] = False,
) -> None:
    """Start a Pioneer LoRA job."""
    with PioneerClient.from_environment() as client:
        result = client.start_training(
            model_name=model_name,
            dataset_name=dataset,
            base_model=base_model,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        if wait:
            result = client.wait_for_training(_required_id(result, "training job"))
    _json(result)


@app.command()
def status(job_id: Annotated[str, typer.Argument()]) -> None:
    """Read a Pioneer training job."""
    with PioneerClient.from_environment() as client:
        _json(client.get_training(job_id))


@app.command()
def evaluate(
    model_id: Annotated[str, typer.Option()],
    dataset: Annotated[str, typer.Option()],
    wait: Annotated[bool, typer.Option()] = False,
) -> None:
    """Start a Pioneer evaluation."""
    with PioneerClient.from_environment() as client:
        started = client.start_evaluation(model_id=model_id, dataset_name=dataset)
        result: Any = started
        if wait:
            if not started.evaluations:
                raise typer.BadParameter("Pioneer did not return an evaluation ID")
            result = client.wait_for_evaluation(started.evaluations[0].id)
    _json(result)


@app.command("evaluation-status")
def evaluation_status(evaluation_id: Annotated[str, typer.Argument()]) -> None:
    """Read a Pioneer evaluation."""
    with PioneerClient.from_environment() as client:
        _json(client.get_evaluation(evaluation_id))


@app.command()
def pipeline(
    artifacts_dir: Annotated[Path, typer.Option()] = Path("training/artifacts/v0"),
    prefix: Annotated[str, typer.Option()] = "cala-fastpath-v0",
    model_name: Annotated[str, typer.Option()] = "cala-fastpath-v0",
    base_model: Annotated[str, typer.Option()] = "fastino/gliner2-multi-v1",
    epochs: Annotated[int, typer.Option(min=1)] = 5,
    learning_rate: Annotated[float, typer.Option(min=0)] = 5e-5,
) -> None:
    """Upload, train, and evaluate base and LoRA models."""
    artifacts = artifacts_dir / "pioneer"
    train_path = artifacts / "train.jsonl"
    validation_path = artifacts / "validation.jsonl"
    if not train_path.exists() or not validation_path.exists():
        raise typer.BadParameter(f"missing built datasets under {artifacts}")

    with PioneerClient.from_environment() as client:
        train_name = f"{prefix}-train"
        evaluation_name = f"{prefix}-evaluation"
        train_dataset = client.upload_dataset(
            train_path,
            dataset_name=train_name,
            purpose="training",
        )
        client.wait_for_dataset(train_dataset.dataset_name)
        evaluation_dataset = client.upload_dataset(
            validation_path,
            dataset_name=evaluation_name,
            purpose="evaluation",
        )
        client.wait_for_dataset(evaluation_dataset.dataset_name)

        training = client.start_training(
            model_name=model_name,
            dataset_name=train_name,
            base_model=base_model,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        training = client.wait_for_training(_required_id(training, "training job"))
        trained_model_id = _required_id(training, "trained model")

        evaluations: dict[str, JsonObject] = {}
        for name, model_id in (("base", base_model), ("lora", trained_model_id)):
            started = client.start_evaluation(
                model_id=model_id,
                dataset_name=evaluation_name,
            )
            if not started.evaluations:
                raise typer.BadParameter(f"Pioneer did not return an evaluation ID for {name}")
            evaluations[name] = client.wait_for_evaluation(started.evaluations[0].id)

    _json(
        {
            "datasets": {
                "train": train_dataset.model_dump(),
                "evaluation": evaluation_dataset.model_dump(),
            },
            "training": training,
            "evaluations": evaluations,
        }
    )


if __name__ == "__main__":
    app()
