from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from arenapilot.agent_contract import AGENT_CONTRACT_VERSION
from arenapilot.cli_entry import app
from arenapilot.workspace import load_experiment_spec


runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "arenapilot"


def _contract() -> dict[str, object]:
    return yaml.safe_load((SKILL_ROOT / "contract.yaml").read_text(encoding="utf-8"))


def _ready_workspace(tmp_path, monkeypatch):
    target = tmp_path / "agent-contract"
    result = runner.invoke(
        app,
        ["init", "kaggle:agent-contract", "--path", str(target), "--json"],
    )
    assert result.exit_code == 0, result.output
    monkeypatch.chdir(target)

    result = runner.invoke(
        app,
        [
            "intake",
            "set",
            "--task",
            "binary_classification",
            "--target",
            "target",
            "--metric",
            "roc_auc",
            "--direction",
            "maximize",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        [
            "validation",
            "configure",
            "val-v1",
            "--split",
            "stratified_kfold",
            "--prediction",
            "probability",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["validation", "activate", "val-v1", "--json"])
    assert result.exit_code == 0, result.output
    return target


def test_skill_contract_matches_runtime_version_and_references() -> None:
    contract = _contract()
    assert contract["schema_version"] == 1
    assert contract["required_runtime_contract_version"] == AGENT_CONTRACT_VERSION == 1
    assert contract["mutation_boundary"] == "arena_cli"

    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert skill.startswith("---\nname: arenapilot\n")
    assert "All runtime mutations go through the `arena` CLI" in skill

    for relative in contract["references"]:
        path = SKILL_ROOT / str(relative)
        assert path.is_file(), relative
        assert str(relative) in skill

    metadata = yaml.safe_load(
        (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    )
    assert metadata["interface"]["display_name"] == "ArenaPilot"
    assert metadata["policy"]["allow_implicit_invocation"] is True


def test_runtime_contract_command_is_machine_readable() -> None:
    result = runner.invoke(app, ["contract", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["contract_version"] == AGENT_CONTRACT_VERSION
    assert payload["mutation_boundary"] == "arena_cli"
    assert "cli_is_the_supported_mutation_boundary" in payload["invariants"]
    assert "arena_managed_kaggle_data_download" in payload["unsupported"]


def test_machine_readable_contract_command_paths_exist() -> None:
    contract = _contract()
    for command_path in contract["public_command_paths"]:
        result = runner.invoke(app, [*command_path, "--help"])
        assert result.exit_code == 0, f"{' '.join(command_path)}\n{result.output}"


def test_exp_configure_keeps_draft_authoring_inside_cli_and_freeze_closes_it(
    tmp_path,
    monkeypatch,
) -> None:
    target = _ready_workspace(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "exp",
            "new",
            "--title",
            "baseline",
            "--hypothesis",
            "A configured baseline establishes the comparison floor.",
            "--model-family",
            "catboost",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    experiment = json.loads(result.output)["experiment"]

    result = runner.invoke(
        app,
        [
            "exp",
            "configure",
            experiment,
            "--model-params-json",
            '{"depth":8,"learning_rate":0.04}',
            "--pipeline-json",
            '{"features":{"frequency_encoding":true}}',
            "--resources-json",
            '{"accelerator":"gpu"}',
            "--backend",
            "kaggle",
            "--seed",
            "7",
            "--tag",
            "categorical",
            "--tag",
            "gpu",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["runtime"]["backend"] == "kaggle"
    assert payload["seed"]["value"] == 7
    assert payload["tags"] == ["categorical", "gpu"]

    from arenapilot.workspace import discover_workspace

    workspace = discover_workspace(target)
    spec = load_experiment_spec(workspace, experiment)
    assert spec.model["params"]["depth"] == 8
    assert spec.pipeline["features"]["frequency_encoding"] is True
    assert spec.runtime.resources["accelerator"] == "gpu"

    result = runner.invoke(app, ["exp", "freeze", experiment, "--json"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "exp",
            "configure",
            experiment,
            "--model-params-json",
            '{"depth":10}',
            "--json",
        ],
    )
    assert result.exit_code == 1
    error = json.loads(result.output)
    assert error["error"]["code"] == "EXPERIMENT_CONFIG_IMMUTABLE"
