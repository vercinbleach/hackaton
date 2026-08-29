from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from cala_fastpath_training import cli
from cala_fastpath_training.models import UploadedDataset

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "training" / "config" / "catalog.json"
EXAMPLES = ROOT / "training" / "data" / "examples.jsonl"
RUNNER = CliRunner()


def _project_root(tmp_path: Path, monkeypatch) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(cli, "ROOT", project)
    return project


def test_generate_rejects_output_outside_project_before_building_generators(
    tmp_path: Path, monkeypatch
) -> None:
    _project_root(tmp_path, monkeypatch)
    outside = tmp_path / "outside.jsonl"

    def unexpected_build(*args, **kwargs):
        raise AssertionError("generators must not be built for an unsafe output path")

    monkeypatch.setattr(cli, "build_generators", unexpected_build)
    result = RUNNER.invoke(cli.app, ["generate", "--output", str(outside)])

    assert result.exit_code == 2
    assert "--output must stay within" in result.output
    assert not outside.exists()


def test_bootstrap_rejects_parent_traversal(tmp_path: Path, monkeypatch) -> None:
    project = _project_root(tmp_path, monkeypatch)
    output = project / "training" / "data" / ".." / ".." / "outside.jsonl"

    result = RUNNER.invoke(
        cli.app,
        ["bootstrap", "--output", str(output), "--catalog", str(CATALOG)],
    )

    assert result.exit_code == 2
    assert "--output must stay within" in result.output
    assert not (project / "outside.jsonl").exists()


def test_build_rejects_absolute_output_directory_outside_project(
    tmp_path: Path, monkeypatch
) -> None:
    _project_root(tmp_path, monkeypatch)
    outside = tmp_path / "artifacts"

    result = RUNNER.invoke(
        cli.app,
        [
            "build",
            str(EXAMPLES),
            "--output-dir",
            str(outside),
            "--catalog",
            str(CATALOG),
        ],
    )

    assert result.exit_code == 2
    assert "--output-dir must stay within" in result.output
    assert not outside.exists()


def test_output_rejects_symbolic_linked_parent(tmp_path: Path, monkeypatch) -> None:
    project = _project_root(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_runs = project / "benchmark" / "runs"
    linked_runs.parent.mkdir(parents=True)
    try:
        linked_runs.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    with pytest.raises(typer.BadParameter, match="symbolic links or junctions"):
        cli._resolve_output_path(linked_runs / "result.jsonl", Path("benchmark/runs"), "--output")


def test_atomic_write_does_not_modify_external_hardlink(tmp_path: Path, monkeypatch) -> None:
    project = _project_root(tmp_path, monkeypatch)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("keep me", encoding="utf-8")
    output = project / "training" / "data" / "generated.jsonl"
    output.parent.mkdir(parents=True)
    try:
        os.link(outside, output)
    except OSError as exc:
        pytest.skip(f"hardlinks are unavailable: {exc}")

    cli._write_output_text(output, "replacement", Path("training/data"), "--output")

    assert output.read_text(encoding="utf-8") == "replacement"
    assert outside.read_text(encoding="utf-8") == "keep me"


def test_allowed_output_paths_still_work(tmp_path: Path, monkeypatch) -> None:
    project = _project_root(tmp_path, monkeypatch)
    generate_output = project / "benchmark" / "runs" / "generated.jsonl"
    bootstrap_output = project / "training" / "data" / "generated.jsonl"
    build_output = project / "training" / "artifacts" / "test-build"

    # Create a valid skill file for the generate command
    skill_file = project / "benchmark" / "skills" / "cala-query" / "SKILL.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_file.write_text("test skill content", encoding="utf-8")

    monkeypatch.setattr(cli, "build_generators", lambda *args, **kwargs: [])
    monkeypatch.setattr(cli, "_read_benchmark_cases", lambda path: [])
    generate_result = RUNNER.invoke(
        cli.app,
        ["generate", "--output", str(generate_output)],
    )
    bootstrap_result = RUNNER.invoke(
        cli.app,
        ["bootstrap", "--output", str(bootstrap_output), "--catalog", str(CATALOG)],
    )
    build_result = RUNNER.invoke(
        cli.app,
        [
            "build",
            str(bootstrap_output),
            "--output-dir",
            str(build_output),
            "--catalog",
            str(CATALOG),
        ],
    )

    assert generate_result.exit_code == 0
    assert bootstrap_result.exit_code == 0
    assert build_result.exit_code == 0
    assert generate_output.read_text(encoding="utf-8") == ""
    assert bootstrap_output.exists()
    assert (build_output / "schema.json").exists()
    assert (build_output / "manifest.json").exists()


def test_generate_rejects_skill_outside_allowed_directory(tmp_path: Path, monkeypatch) -> None:
    """Reject skill paths that point outside benchmark/skills."""
    _project_root(tmp_path, monkeypatch)
    outside = tmp_path / "outside.md"
    outside.write_text("sensitive data", encoding="utf-8")

    monkeypatch.setattr(cli, "_read_benchmark_cases", lambda path: [])
    result = RUNNER.invoke(
        cli.app,
        ["generate", "--skill", str(outside), "--systems", "openai-skill"],
    )

    assert result.exit_code == 2
    assert "must stay within" in result.output


def test_generate_rejects_skill_with_path_traversal(tmp_path: Path, monkeypatch) -> None:
    """Reject skill paths using ../ to escape the allowed directory."""
    project = _project_root(tmp_path, monkeypatch)
    outside = tmp_path / "outside.md"
    outside.write_text("sensitive data", encoding="utf-8")

    # Create skills directory
    skills_dir = project / "benchmark" / "skills"
    skills_dir.mkdir(parents=True)

    # Try to use path traversal
    traversal = skills_dir / ".." / ".." / ".." / "outside.md"

    monkeypatch.setattr(cli, "_read_benchmark_cases", lambda path: [])
    result = RUNNER.invoke(
        cli.app,
        ["generate", "--skill", str(traversal), "--systems", "openai-skill"],
    )

    assert result.exit_code == 2
    assert "must stay within" in result.output


def test_generate_rejects_skill_symlink(tmp_path: Path, monkeypatch) -> None:
    """Reject skill files that are symlinks to outside files."""
    project = _project_root(tmp_path, monkeypatch)
    outside = tmp_path / "outside.md"
    outside.write_text("sensitive data", encoding="utf-8")

    skills_dir = project / "benchmark" / "skills"
    skills_dir.mkdir(parents=True)
    link = skills_dir / "link.md"

    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    monkeypatch.setattr(cli, "_read_benchmark_cases", lambda path: [])
    result = RUNNER.invoke(
        cli.app,
        ["generate", "--skill", str(link), "--systems", "openai-skill"],
    )

    assert result.exit_code == 2
    assert "symbolic links or junctions" in result.output


def test_generate_rejects_skill_hardlink(tmp_path: Path, monkeypatch) -> None:
    """Reject skill files that are hardlinks to outside files."""
    project = _project_root(tmp_path, monkeypatch)
    outside = tmp_path / "outside.md"
    outside.write_text("sensitive data", encoding="utf-8")

    skills_dir = project / "benchmark" / "skills"
    skills_dir.mkdir(parents=True)
    hardlink = skills_dir / "hardlink.md"

    try:
        os.link(outside, hardlink)
    except OSError as exc:
        pytest.skip(f"hardlinks are unavailable: {exc}")

    monkeypatch.setattr(cli, "_read_benchmark_cases", lambda path: [])
    result = RUNNER.invoke(
        cli.app,
        ["generate", "--skill", str(hardlink), "--systems", "openai-skill"],
    )

    assert result.exit_code == 2
    assert "multiple hard links" in result.output


def test_generate_rejects_skill_directory(tmp_path: Path, monkeypatch) -> None:
    """Reject skill paths that point to directories."""
    project = _project_root(tmp_path, monkeypatch)
    skills_dir = project / "benchmark" / "skills"
    skills_dir.mkdir(parents=True)
    directory = skills_dir / "subdir"
    directory.mkdir()

    monkeypatch.setattr(cli, "_read_benchmark_cases", lambda path: [])
    result = RUNNER.invoke(
        cli.app,
        ["generate", "--skill", str(directory), "--systems", "openai-skill"],
    )

    assert result.exit_code == 2
    assert "must be a file" in result.output


def test_generate_accepts_valid_skill_file(tmp_path: Path, monkeypatch) -> None:
    """Accept a valid skill file within benchmark/skills."""
    project = _project_root(tmp_path, monkeypatch)
    skills_dir = project / "benchmark" / "skills"
    skills_dir.mkdir(parents=True)
    skill_file = skills_dir / "valid.md"
    skill_file.write_text("valid skill content", encoding="utf-8")

    monkeypatch.setattr(cli, "build_generators", lambda *args, **kwargs: [])
    monkeypatch.setattr(cli, "_read_benchmark_cases", lambda path: [])
    result = RUNNER.invoke(
        cli.app,
        ["generate", "--skill", str(skill_file), "--systems", "openai-skill"],
    )

    assert result.exit_code == 0


def test_secure_read_artifact_rejects_symlink(tmp_path: Path) -> None:
    """Reject artifact files that are symlinks."""
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b'{"sensitive": "data"}')

    link = artifacts_root / "train.jsonl"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    with pytest.raises(typer.BadParameter, match="symbolic links or junctions"):
        cli._secure_read_artifact(link, artifacts_root, "training dataset")


def test_secure_read_artifact_rejects_hardlink(tmp_path: Path) -> None:
    """Reject artifact files that are hardlinks."""
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b'{"sensitive": "data"}')

    hardlink = artifacts_root / "train.jsonl"
    try:
        os.link(outside, hardlink)
    except OSError as exc:
        pytest.skip(f"hardlinks are unavailable: {exc}")

    with pytest.raises(typer.BadParameter, match="multiple hard links"):
        cli._secure_read_artifact(hardlink, artifacts_root, "training dataset")


def test_secure_read_artifact_rejects_path_outside_root(tmp_path: Path) -> None:
    """Reject artifact files outside the allowed root."""
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b'{"sensitive": "data"}')

    with pytest.raises(typer.BadParameter, match="must stay within"):
        cli._secure_read_artifact(outside, artifacts_root, "training dataset")


def test_secure_read_artifact_accepts_valid_file(tmp_path: Path) -> None:
    """Accept a valid artifact file within the allowed root."""
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    valid_file = artifacts_root / "train.jsonl"
    valid_file.write_bytes(b'{"valid": "data"}')

    content = cli._secure_read_artifact(valid_file, artifacts_root, "training dataset")
    assert content == b'{"valid": "data"}'


def test_pipeline_uses_unique_dataset_names(tmp_path: Path, monkeypatch) -> None:
    project = _project_root(tmp_path, monkeypatch)
    artifacts = project / "training" / "artifacts" / "v0" / "pioneer"
    artifacts.mkdir(parents=True)
    (artifacts / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (artifacts / "validation.jsonl").write_text("{}\n", encoding="utf-8")
    uploaded_names: list[str] = []

    class FakePioneer:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def upload_dataset(self, path, *, dataset_name, purpose, content=None):
            uploaded_names.append(dataset_name)
            return UploadedDataset(
                dataset_id=f"id-{purpose}",
                dataset_name=dataset_name,
                version_number="1",
            )

        def wait_for_dataset(self, dataset):
            return {"status": "ready"}

        def start_training(self, **kwargs):
            return {"id": "training-1"}

        def wait_for_training(self, job_id):
            return {"id": "model-1", "status": "complete"}

        def start_evaluation(self, **kwargs):
            return SimpleNamespace(evaluations=[SimpleNamespace(id="evaluation-1")])

        def wait_for_evaluation(self, evaluation_id):
            return {"status": "complete"}

    fake = FakePioneer()
    monkeypatch.setattr(cli.secrets, "token_hex", lambda size: "unique-run")
    monkeypatch.setattr(cli.PioneerClient, "from_environment", lambda: fake)

    result = RUNNER.invoke(
        cli.app,
        ["pipeline", "--artifacts-dir", str(project / "training" / "artifacts" / "v0")],
    )

    assert result.exit_code == 0
    assert uploaded_names == [
        "cala-fastpath-v0-unique-run-train",
        "cala-fastpath-v0-unique-run-evaluation",
    ]
