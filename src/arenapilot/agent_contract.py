from __future__ import annotations


AGENT_CONTRACT_VERSION = 1

CAPABILITIES = [
    "workspace_bootstrap",
    "competition_intake",
    "validation_activation",
    "experiment_authoring",
    "experiment_freeze_and_lineage",
    "local_execution",
    "kaggle_remote_execution",
    "artifact_verification",
    "mlflow_tracking",
    "experiment_comparison",
    "submission_lifecycle",
    "cross_competition_memory",
    "knowledge_promotion",
]

INVARIANTS = [
    "experiment_is_intent_run_is_execution",
    "frozen_experiment_specs_are_immutable",
    "validation_versions_are_immutable_after_activation",
    "comparison_requires_matching_comparison_domain",
    "every_execution_creates_a_new_run",
    "process_success_does_not_imply_verified_run",
    "remote_jobs_do_not_write_directly_to_mlflow",
    "submissions_require_verified_runs",
    "memory_claims_require_evidence",
    "contradictory_evidence_is_preserved",
    "high_confidence_requires_explicit_approval",
    "cli_is_the_supported_mutation_boundary",
]

READ_ONLY_COMMANDS = [
    "arena version",
    "arena contract --json",
    "arena status --json",
    "arena doctor --json",
    "arena exp show <exp> --json",
    "arena exp list --json",
    "arena exp compare <baseline> <candidate> --json",
    "arena exp lineage <exp> --json",
    "arena run show <run> --json",
    "arena run list --json",
    "arena run logs <run>",
    "arena remote status <run> --json",
    "arena remote logs <run>",
    "arena submit status <submission> --json",
    "arena submit budget --json",
    "arena submissions --json",
    "arena fingerprint show --json",
    "arena evidence list --json",
    "arena finding show <finding> --json",
    "arena finding list --json",
    "arena failure list --json",
    "arena knowledge retrieve --json",
    "arena knowledge show <type:key> --json",
    "arena knowledge assess <type:key> --json",
    "arena knowledge history <type:key> --json",
    "arena knowledge ranked --json",
    "arena technique list --json",
]

MUTATING_COMMANDS = [
    "arena init kaggle:<slug>",
    "arena intake set ...",
    "arena validation configure <validation> ...",
    "arena validation activate <validation>",
    "arena exp new ...",
    "arena exp configure <exp> ...",
    "arena exp freeze <exp>",
    "arena exp run <exp> [--backend local|kaggle]",
    "arena run verify <run>",
    "arena remote recover <run>",
    "arena submit create --run <run> [--file <csv>]",
    "arena submit validate <submission>",
    "arena submit send <submission>",
    "arena fingerprint set ...",
    "arena evidence compare ...",
    "arena evidence note ...",
    "arena finding create ...",
    "arena finding approve <finding>",
    "arena learn [--finding <finding>]",
    "arena independence set ...",
    "arena technique register <key> ...",
    "arena technique deprecate <key> ...",
    "arena knowledge approve <type:key> ...",
    "arena knowledge deprecate <type:key> ...",
]

UNSUPPORTED = [
    "automatic_kaggle_competition_metadata_inspection",
    "arena_managed_kaggle_data_download",
    "creating_validation_v2_plus_from_cli",
    "automatic_hyperparameter_search",
    "automatic_submission_without_explicit_send_command",
    "direct_provider_api_calls_from_the_skill",
    "direct_sqlite_or_mlflow_mutation_from_the_skill",
]


def contract_payload() -> dict[str, object]:
    return {
        "ok": True,
        "contract_version": AGENT_CONTRACT_VERSION,
        "mutation_boundary": "arena_cli",
        "capabilities": CAPABILITIES,
        "invariants": INVARIANTS,
        "read_only_commands": READ_ONLY_COMMANDS,
        "mutating_commands": MUTATING_COMMANDS,
        "unsupported": UNSUPPORTED,
    }
