from __future__ import annotations

import os
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from cala_fastpath_training import cli

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
